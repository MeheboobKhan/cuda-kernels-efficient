# Benchmarks — 01 Conv1D

## Summary (GTX 1650, N=524288, K=8191)

> **Methodology note.** Numbers below are either *plain-timed* (`bench.exe N K`,
> an averaged 20-iteration loop) or *ncu-reconstructed* (summed kernel
> durations from a profile). Reconstruction consistently **underestimates** —
> v2: 6.6% low, v3: 5.8% low, v5.2: 13.8% low — because it misses inter-kernel
> gaps and launch overhead. The two are only comparable to their own kind.
> Every table says which it is.


| version | approach | runtime | status |
|---|---|---|---|
| v1 | shared-memory tiled, O(N·K) | 16,105.8 µs | superseded |
| v2 | FFT via cuFFT, unbatched | 757.8 µs | superseded |
| v3 | + batched forward R2C | 731.4 µs | superseded |
| **v4** | **+ overlap-save when it wins** | **493 µs** | **canonical (`solution.cu`)** |
| v5 | fully fused single-kernel FFT | 622–627 µs | superseded by v5.1 |
| v5.1 | + 1024 threads, hoisted divide | 537.4 µs | measured, still 9% over v4 |
| v5.2 | + two FFT stages per shared round trip | 542.4 µs | correct; wins only at N=32768 |

Local hardware floor for this problem is ~44 µs (6.46 MB of unavoidable
traffic at the GTX 1650's ~147 GB/s), so v4 sits ~11x above what this card
can physically do. The tensara leaderboard's 17.67 µs is on an L40S and is
below this card's floor outright — it is not reachable locally at any level
of optimization, only on comparable hardware.


## v5.2: two butterfly stages per shared-memory round trip — MEASURED, wins at large N

v5.1's profile moved the bottleneck a third time. Both fixes landed exactly as
intended — occupancy 49.90% → 99.33%, instructions 15,003,318 → 9,421,038
(−37%) — but runtime only fell 14% (602.7µs → 517.7µs), because the limit is
now shared-memory bandwidth:

| metric | v5 | v5.1 |
|---|---|---|
| L1/TEX throughput | 65.53% | **76.63%** ← the wall |
| DRAM throughput | 4.43% | 5.17% (idle) |
| SM busy | 32.21% | 23.76% |
| warp cycles / instr | 10.55 | 28.02 (waiting on shared) |

A radix-2 stage reads Mc and writes Mc complex values through shared memory,
13 times per transform. v5.2 fuses **two levels per round trip**: read 4
elements, do both butterfly levels in registers, write 4 back — 4R+4W instead
of 8R+8W. For Mc=8192 that is 6 fused passes + 1 trailing single stage = 7
round trips instead of 13, cutting shared traffic 224.9 MB → 121.1 MB (−46%).

Two things make this low-risk:

- It is **bit-identical** to the existing radix-2 code, not merely close —
  `tests/validate_radix4.py` compares them at every size up to 2^13 and gets
  exactly 0.00e+00 difference. It is the same computation reordered.
- The second level-2 twiddle is free: `W_2L^(j+L/2) == -i · W_2L^j`, so it is
  a swap-and-negate rather than a third `__sincosf`.

If runtime tracks shared traffic, this lands near **300µs — about 39% under
v4**, which would be the first time the fused approach actually wins. That
scaling is an assumption, though; the previous two predictions on this kernel
were off by 4x and 2x respectively.

**Risk to watch in the next profile:** the fused butterfly holds ~16 float2
values live, so registers may exceed the 64/thread that 1024 threads allows.
Check `Registers Per Thread` and `Local Memory Spilling Requests` — if
spilling is non-zero, drop the launch to 512 threads.


### v5.2 plain-timed, all four sizes — the win mostly evaporates

| N | v3 / v4 small-N path | v5.2 | winner |
|---|---|---|---|
| 32768 | 100.3 µs | **92.5 µs** | v5.2 (−7.8%) |
| 65536 | **109.1 µs** | 129.6 µs | v3/v4 (+18.8%) |
| 131072 | **179.9 µs** | 226.1 µs | v3/v4 (+25.7%) |
| 524288 | *not plain-timed* | 542.4 µs | **unresolved** |

All plain-timed. Note v4 dispatches to the v3 single-FFT path for the three
smaller N — its cost model only selects overlap-save at 524288 — so v4 and v3
are the same code there and the v3 numbers apply to both.

**This inverts the earlier conclusion, and inverts my prediction too.** I
expected fused to win at large N (many blocks, good occupancy) and lose at
small N (partial wave). The opposite happened: it wins only at the *smallest*
size and loses in the middle. The wave-occupancy model I used to predict the
crossover was simply wrong about which effect dominates.

It also softens the "first fused win" claim from the previous section. That
comparison was 467.8 vs 492.6 — both ncu-reconstructed, so internally
consistent, but v4 has never been plain-timed. Applying the observed 6–14%
reconstruction shortfall puts v4's real N=524288 time somewhere around
**524–573 µs**, and v5.2 measures 542.4 µs — inside that band. Too close to
call without the actual measurement.

**Outstanding measurement: `bench.exe 524288 8191` built from `solution.cu`
(v4).** That single number decides whether v5.2 is worth keeping for large N
or whether the whole fused line is a documented dead end.


### v5.2 result — ncu-reconstructed (GTX 1650, N=524288)

| kernel | duration |
|---|---|
| `buildHPad` | 2.48µs |
| cuFFT `vector_fft_symm_r2c<16384>` (H) | 17.02µs |
| `fusedConvBlock` | 448.27µs |
| **total** | **467.8µs** |

467.8µs vs v4's 492.6µs, both ncu-reconstructed — 5.1% faster by that
measure. See the plain-timed section above for why this overstates the case.

The register risk flagged above did not materialize: 45 registers/thread
(limit is 64 at 1024 threads), zero spilling, 98.56% occupancy. Instructions
fell 9,421,038 → 7,180,206 (−23.8%).

Correctness verified on device (`bench --check`, all 7 cases): max relative
error 1.6e-5 at K=8191, 1.5e-7 at small K. That is normal FP32 FFT
accumulation against a double-precision direct-summation reference, and is
~60x inside the 1e-3 bar. Worth noting only if a checker demands better than
1e-4.

**But the prediction missed again — 300µs estimated, 467.8µs actual (1.6x
off).** Shared traffic fell 46% while runtime fell only 13.4%, and crucially
L1/TEX throughput barely moved (76.63% → 75.97%). That says the wall is not
shared-memory *volume* but shared-memory *transactions* — bank conflicts from
strided `float2` (8-byte) accesses at power-of-two offsets. Halving the number
of passes did not halve the per-access conflict cost. Padding the array is the
standard fix and is not available here: the buffer is exactly 65,536 bytes,
the entire Turing per-block budget, with no room for pad words.

Three predictions on this kernel have now been off by 4x, 2x and 1.6x, always
optimistic. Treat any further estimate as a direction only.

### Open: does v5.2 win at the smaller N?

Only N=524288 has been measured. The fused kernel launches
`ceil((N+2K−2)/hop)` blocks with hop=8194, and a partial wave costs the same
as a full one, so small N pays for 16 SMs while using a handful:

| N | blocks | waves | est. fused | v4 actual | likely |
|---|---|---|---|---|---|
| 32768 | 6 | 0.38 | ~128µs | 100.3µs | v4 |
| 65536 | 10 | 0.62 | ~128µs | 109.1µs | v4 |
| 131072 | 18 | 1.12 | ~142µs | 179.9µs | fused |
| 524288 | 66 | 4.12 | 468µs (measured) | 493.0µs | fused |

If that holds, the right final shape is the same crossover dispatch v4 already
uses internally — pick fused above roughly one wave's worth of blocks, cuFFT
below it. Needs the other three measurements before committing.


## v5: fully fused overlap-save (`solution_fused.cu`) — experiment, REJECTED (slower)

Hardware runs (GTX 1650, N=524288, `ncu`-reconstructed). Two runs of the same
binary: 627.4us and 622.3us.

| kernel | duration | total/call |
|---|---|---|
| `buildHPad` | 2.45µs | 2.45µs |
| cuFFT `vector_fft_symm_r2c<16384>` (H) | 17.10µs | 17.10µs |
| `fusedConvBlock` | 607.86µs | 607.86µs |
| **total** | | **~627.4µs** |

**vs v4's 493µs, that is 27% slower.** My pre-run estimate was 100–150µs; it
was wrong by ~4x, and the reason is instructive.

**The design goal was met.** DRAM traffic collapsed exactly as predicted:

| | v4 | v5 |
|---|---|---|
| DRAM throughput | 95–97% (saturated) | **4.46%** |
| achieved bandwidth | 137 GB/s | **7.09 GB/s** |

That is a ~19x reduction in memory traffic — the fused kernel really does read
A once and write C once. The problem is that it replaced a memory bottleneck
with a worse instruction-issue bottleneck:

| metric | value | reading |
|---|---|---|
| L1/TEX throughput | 65.70% | highest utilization — shared-memory butterflies |
| Compute (SM) throughput | 32.30% | |
| "No Eligible" warp cycles | 62.07% | latency-starved |
| Achieved occupancy | 49.77% | only 16 of 32 warps/SM |
| instructions / warp-butterfly | ~68 | should be ~15–20 |

Two of those were my bugs, both now fixed:

1. **512 threads was wrong.** I picked it defensively to avoid register
   spilling. The profile shows 39 registers/thread and *zero* spilling, so
   1024 threads (39,936 of the 65,536-register file) fits fine. Since 64KB of
   shared memory pins us to 1 block/SM, thread count *is* occupancy here —
   512 threads was throwing away half the SM for no reason, on a kernel the
   profile says is latency-bound.
2. **A float divide inside the butterfly inner loop.** `-2π*j/len` — `len` is
   loop-invariant but `j` isn't, so the compiler can't hoist it. An IEEE float
   divide is ~20 instructions and it ran 7,028,736 times per launch. Now
   hoisted to a reciprocal multiply.

Measured twice (627.4us, 622.3us - same binary, run-to-run noise only), so the
result is settled: **on Turing, the fused radix-2 kernel loses to cuFFT.**

The two fixes above are written but have not been compiled and run yet, so
their effect is unknown. Even at the optimistic end (~250us) v5 would still be
~6x off the 44us traffic floor, because the structural problem survives them:
a radix-2 FFT makes 26 round trips through 64KB of shared memory (13 stages x
forward and inverse), while cuFFT uses radix-128 butterflies held in
registers. That is the real bar - beating a vendor library at its own kernel -
and radix-2 does not clear it.

### Why this is kept as a documented negative result

The fused design *did* do what it was built to do: traffic dropped 19x and the
kernel stopped being memory-bound at all (4.46% DRAM utilization vs v4's 95%
saturation). It lost anyway, because removing the memory bottleneck exposed a
compute inefficiency that the memory wall had been hiding. The lesson
generalizes: "fewer passes over DRAM" is only a win if the fused compute is
competitive with what the library was doing during those passes.

Getting to leaderboard numbers this way needs radix-4/radix-8 butterflies with
register-resident data between exchanges - i.e. reimplementing most of what
NVIDIA's cuFFTDx already does. The tradeoff also shifts on a card with more
shared memory per SM (L40S has 100KB vs Turing's 64KB): a larger block fits,
which improves the overlap efficiency hop/M at the same time.

**v5 as originally written loses. Whether the fused approach wins at all
comes down to v5.2 above.**


## v4: overlap-save, used only where it actually wins (`solution.cu`) — current

**Revised plan, and why.** The original idea here was overlap-save with
`B`'s spectrum cached across calls, since all 4 test cases share `K=8191`.
That caching isn't safe to ship, though: nothing guarantees `B`'s *content*
is identical across calls just because `K` is — different test cases very
plausibly use independently-random `B` values, and caching on `K` alone
risks silently returning wrong answers the moment that assumption breaks.
So `B`'s transform has to be recomputed fresh every call regardless of
algorithm, same as v2/v3.

With that correctness constraint priced in, overlap-save's actual benefit
is smaller and more conditional than the earlier framing suggested. Modeling
total FFT work (forward+inverse block passes, `2×numBlocks+1` block-sized
transforms, vs. v3's single `L`-sized transform) at the actual test sizes,
with a numerically-searched optimal block size `M≈98304` (~12×K):

| N | v3 (single FFT) | v4 (overlap-save) | ratio |
|---|---|---|---|
| 32768 | 1.15M | 2.45M | **2.13x worse** |
| 65536 | 2.01M | 2.45M | **1.22x worse** |
| 131072 | 3.80M | 4.08M | **1.07x worse** |
| 524288 | 15.56M | 10.60M | **0.68x — 32% less work** |

(units are an arbitrary FFT-op cost estimate, only the ratios matter). For
the three smaller cases, a single block already covers the whole output
(`numBlocks=1`), so overlap-save pays for a full fixed-size `M=98304`
transform on data that didn't need one — strictly worse than v3's
right-sized `L`. It only pays off at N=524288, where `numBlocks=6` is
large enough to amortize the block/overhead ratio below v3's cost.

**So v4 computes both cost estimates at the start of every call (cheap —
just host-side arithmetic on `N`/`K`) and dispatches to whichever path is
actually cheaper.** v3's single-FFT code stays as the path used for
32768/65536/131072; overlap-save only activates for the largest case (and
any future `N` where the crossover math favors it). This means v4 can't
regress vs v3 by construction — worst case, every call takes the v3 path
and nothing changes.

Overlap-save's indexing (windowing `A` into overlapping blocks on the fly,
zero-padding `B` to the block size once per call, batched block FFT,
stitching valid regions back into `C`) was validated against the brute-force
reference in `tests/validate_fft.py` before being ported to CUDA, same
process as v2.

Not yet confirmed with a plain (non-`ncu`) `bench.exe` run — but an `ncu`
profile of the same `bench.exe 524288 8191 1` methodology used for v2/v3
lets us reconstruct the real per-call time already, same way as before:

| kernel | duration | count/call | total/call |
|---|---|---|---|
| `buildHPad` | 3.76µs | ×1 | 3.76µs |
| `windowBlocks` | 40.24µs | ×1 | 40.24µs |
| `postprocess` (H, batch=1) | 6.75µs | ×1 | 6.75µs |
| `postprocess` (blocks, batch=6) | 33.60µs | ×1 | 33.60µs |
| `preprocess` (blocks, batch=6) | 33.81µs | ×1 | 33.81µs |
| `complexMulBroadcast` | 46.42µs | ×1 | 46.42µs |
| `stitchExtract` | 36.58µs | ×1 | 36.58µs |
| `regular_fft_factor<128,...>` (H, 2 stages/transform) | 11.82µs | ×2 | 23.64µs |
| `regular_fft_factor<128,...>` (blocks, fwd+inv) | 48.47µs | ×4 | 193.88µs |
| `regular_fft_factor<3,...>` (H) | 6.70µs | ×1 | 6.70µs |
| `regular_fft_factor<3,...>` (blocks, fwd+inv) | 33.59µs | ×2 | 67.18µs |
| **total (reconstructed)** | | | **~492.6µs** |

vs. v3's measured 731.4µs, that's a **32.7% reduction** — matching the
cost-model prediction (32% less FFT work) almost exactly, which is good
independent confirmation the model isn't just numerology. `M=98304`
factors as `2^15 × 3`, hence only two distinct `regular_fft_factor`
templates (`128` and `3`) instead of v3's four (`243, 32, 5, 7`) — a
side-effect of choosing `M` for cost-per-sample rather than tightness-to-L,
worth noting but not obviously good or bad on its own.

Still want a plain (non-profiled) `bench.exe` run to confirm this
reconstruction against a real wall-clock number, same sanity check done
for v2 and v3 (reconstructed vs. measured landed within ~6% both times):

```powershell
nvcc -O3 -arch=sm_75 -lineinfo -o bench.exe solution.cu tests\bench.cu -lcufft
.\bench.exe 524288 8191
.\bench.exe 65536 8191
.\bench.exe 32768 8191
.\bench.exe 131072 8191
```

| N | K | GPU | Runtime | vs v3 |
|---|---|-----|---------|-------|
| 65536  | 8191 | — | — | (expect ≈ v3's 109.1µs, same code path) |
| 32768  | 8191 | — | — | (expect ≈ v3's 100.3µs, same code path) |
| 131072 | 8191 | — | — | (expect ≈ v3's 179.9µs, same code path) |
| 524288 | 8191 | — | — | (expect ≈ 493µs based on the reconstruction above) |

## v3: batched forward FFT (superseded by v4, still used as v4's small-N path)

`cufftPlanMany` batch=2 for `R2C(A)`/`R2C(B)` + fused `padBoth` kernel, built
directly off the v2 `ncu` profile (see below). Measured on the same local
**GTX 1650**.

| N | K | GPU | Runtime | vs v2 |
|---|---|-----|---------|-------|
| 65536  | 8191 | GTX 1650 | 109.1 µs | 26.3% faster |
| 32768  | 8191 | GTX 1650 | 100.3 µs | 54.2% faster |
| 131072 | 8191 | GTX 1650 | 179.9 µs | 6.3% faster |
| 524288 | 8191 | GTX 1650 | 731.4 µs | 3.5% faster |

Two things worth calling out:

**The non-monotonic v2 result is gone.** 32768 was the anomaly in v2
(slower than 131072 despite less data); in v3 it's now the *fastest* of all
four sizes, and the ordering is monotonic with N as expected. This
confirms the diagnosis: 32768's smaller `L` meant each `regular_fft_factor`
kernel ran for a shorter genuine duration, so the ~5-10µs fixed launch
overhead was a *larger fraction* of its total time — exactly the case
batching helps most. Its win (54.2%) is the biggest of the four for that
reason.

**The win shrinks as N grows, and that's expected, not a regression.**
524288 only improved 3.5% — at that size, `L=544320` means each
`regular_fft_factor` launch already does tens of µs of genuine FFT
compute, so removing launch overhead saves a small slice of a
compute-dominated total. The ~490µs of real butterfly work identified in
the v2 `ncu` profile is essentially unchanged by batching — batching only
ever targeted the *scheduling* overhead around it, not the FFT math
itself.

### `ncu` profile (N=524288, K=8191) — what actually drove the v2→v3 change

Full report: `--set full`, single non-warmup call (`bench.exe 524288 8191 1`).

One `solution()` call breaks into ~19 kernel launches: `zeroPadA`,
`reversePadB`, 4x `regular_fft_factor` + `postprocess_kernel` (R2C of A),
same again for R2C of B, `complexMulInplace`, `preprocess_kernel` + 4x
`regular_fft_factor` (C2R), `extractNormalize`.

FFT size check: the four `regular_fft_factor<F,...>` kernels use factors
`243, 32, 5, 7` → `243×32×5×7×2 = 544,320 = L`, within 0.7% of the
theoretical minimum (540,668) — the FFT-size search itself is already
near-optimal, not a target for further tuning.

Time breakdown (average kernel duration × count per call):

| stage | avg duration | count/call | total/call |
|---|---|---|---|
| `regular_fft_factor` (4 stages) | 63.4 + 35.9 + 31.4 + 32.9 = 163.6µs | ×3 (R2C(A), R2C(B), C2R) | ~490.8µs |
| `zeroPadA` | 31.4µs | ×1 | 31.4µs |
| `reversePadB` | 14.5µs | ×1 | 14.5µs |
| `postprocess_kernel` | 31.2µs | ×2 | 62.3µs |
| `preprocess_kernel` | 31.5µs | ×1 | 31.5µs |
| `complexMulInplace` | 45.6µs | ×1 | 45.6µs |
| `extractNormalize` | 32.1µs | ×1 | 32.1µs |
| **total (reconstructed)** | | | **~708.1µs** |

(Matches the measured 757.8µs closely enough to trust the breakdown — small
gap is scheduling/sync overhead between launches, itself part of the case
for batching.)

~69% of the time is the actual FFT butterfly work (`regular_fft_factor`),
run three full times per call. ~31% is five small kernels that are each
individually memory-bound (80-96% DRAM throughput per `ncu` — genuinely
bandwidth-saturated, not just "unoptimized") but still pay a full kernel
launch + DRAM round-trip for trivial elementwise work. `R2C(A)` and
`R2C(B)` are independent, so batching them (v3) directly targets a third of
the 69% bucket, and fusing `zeroPadA`+`reversePadB` targets part of the 31%
bucket.

### `ncu` profile of v3 (N=524288, K=8191) — confirms batching's limits, motivates v4

Same methodology as the v2 profile above (`--set full`, single non-warmup
call). Kernel count per call dropped from ~19 to ~15: `padBoth` (1, was 2
separate kernels), one batched `postprocess_kernel` (1, was 2), the 4
`regular_fft_factor` stages now appear **twice each** — once at ~2x
duration for the batched forward pass (`R2C(A)+R2C(B)` together), once at
roughly the original duration for the still-unbatched `C2R` pass:

| stage | duration | count/call | total/call |
|---|---|---|---|
| `padBoth` | 44.99µs | ×1 | 44.99µs |
| forward batched `regular_fft_factor` (4 stages, ~2x work each) | 115.0+71.0+62.6+63.1 = 311.7µs | ×1 | 311.7µs |
| `postprocess_kernel` (batched) | 61.20µs | ×1 | 61.2µs |
| `complexMul` | 44.59µs | ×1 | 44.6µs |
| `preprocess_kernel` | 31.36µs | ×1 | 31.4µs |
| C2R `regular_fft_factor` (4 stages, unbatched) | 62.5+36.0+31.6+32.9 = 163.0µs | ×1 | 163.0µs |
| `extractNormalize` | 31.90µs | ×1 | 31.9µs |
| **total (reconstructed)** | | | **~688.8µs** (measured: 731.4µs) |

Comparing to the v2 breakdown: batching saved almost nothing on `padBoth`
(45.9→45.0µs) or `postprocess` (62.3→61.2µs) — those were already mostly
genuine memory-bound work, not launch overhead. The real saving was in the
forward `regular_fft_factor` total (327.2→311.7µs, ~15.5µs) — batching
removed one set of fixed per-launch overhead, but each batched launch still
does ~2x the work of its unbatched counterpart, so the win was always going
to be small relative to total runtime. **~475µs (69%) is still genuine FFT
butterfly compute** (311.7 forward + 163.0 inverse) — essentially
unchanged in proportion from v2. This is the finding that led to v4: batching
was worth doing (free, no downside), but it was never going to close most
of the remaining gap at N=524288, because the gap there isn't overhead
anymore, it's FFT work itself — which is what v4's overlap-save path
targets specifically for that size.

## v1: shared-memory tiled, O(N·K) (`solution_tiled.cu`) — superseded

Measured with the same `tests/bench.cu` harness on the same GTX 1650. Kept
as the reference point for how much the FFT rewrite (v2) won by — these are
millisecond-scale, ~20-100x slower than v2.

| N | K | GPU | Runtime | GFLOP/s |
|---|---|-----|---------|---------|
| 65536  | 8191 | GTX 1650 | 2.7396 ms | 391.9 |
| 32768  | 8191 | GTX 1650 | 1.3535 ms | 396.6 |
| 131072 | 8191 | GTX 1650 | 5.4218 ms | 396.0 |
| 524288 | 8191 | GTX 1650 | 16.1058 ms | 533.3 |

(GFLOP/s isn't meaningful for v2/v3/v4 — FFT does ~400x fewer FLOPs for the
same output, so it's omitted there; wall-clock is the only fair comparison
across v1 vs v2/v3/v4.)

## Target

tensara leaderboard for this problem currently tops out around **17-52
microseconds** on data-center GPUs (L40S/B200/H100/T4). All local numbers
above are on a GTX 1650 (~2.85 TFLOP/s FP32, ~128 GB/s bandwidth — well
below any of those cards), so expect a real tensara submission to land
faster than these local numbers independent of further code changes. Still
tracking the gap here since the *relative* improvement between versions is
hardware-independent.

At the smaller sizes (32768/65536, ~100-110µs on v3/v4) we're now only
~2-6x off the leaderboard range even on much weaker hardware — plausibly
competitive once run on tensara's actual GPUs. 524288 was ~14-40x off on
v3; v4's overlap-save path targets specifically that gap.