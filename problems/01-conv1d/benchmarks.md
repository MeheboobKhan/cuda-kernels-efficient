# Benchmarks — 01 Conv1D

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