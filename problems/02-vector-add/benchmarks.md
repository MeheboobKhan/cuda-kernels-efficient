# Benchmarks — 02 Vector Addition

## Summary (GTX 1650)

> **Methodology note.** All numbers are plain-timed (`bench.exe N [iters]`,
> a 20-iteration averaged `cudaEvent` loop, 5 iterations at the two
> largest N to keep run time reasonable). GB/s = `3*N*4 bytes / time`
> (2 reads + 1 write, the minimum possible traffic for this problem — no
> version does any extra passes over the data).

| N | v1 (naive) | v2 (float4) | v3 (float4+ILP4, buggy) | v4 (float4+ILP4, fixed) | winner |
|---|---|---|---|---|---|
| 2^20 (1,048,576)   | 143.30 GB/s | 140.76 GB/s | 116.14 GB/s | 142.19 GB/s | v1/v2/v4 tie |
| 2^22 (4,194,304)   | 147.57 GB/s | 147.53 GB/s | 121.53 GB/s | 147.07 GB/s | v1/v2/v4 tie |
| 2^23 (8,388,608)   | 148.17 GB/s | 148.29 GB/s | 126.86 GB/s | 147.70 GB/s | v1/v2/v4 tie |
| 2^25 (33,554,432)  | 148.98 GB/s | 148.85 GB/s | 125.81 GB/s | 148.11 GB/s | v1/v2/v4 tie |
| 2^26 (67,108,864)  | 148.97 GB/s | 148.70 GB/s | 126.44 GB/s | 147.98 GB/s | v1/v2/v4 tie |
| 2^27 (134,217,728) | 146.08 GB/s | 145.60 GB/s | 127.76 GB/s | *pending* | — |
| 2^28 (268,435,456) | 146.23 GB/s | 145.63 GB/s | 126.65 GB/s | *pending* | — |
| 2^29 (536,870,912) | 33.69 GB/s  | 34.43 GB/s  | 7.15 GB/s  | *pending* | — |
| 2^30 (1,073,741,824) | 27.25 GB/s | 27.20 GB/s | 5.51 GB/s | *pending* | — |

**v2 (float4 vectorization) does not beat v1 at any measured size** — every
difference is within normal run-to-run noise (≤2%). See `README.md` for
why: v1 was already bandwidth-saturated at ~148 GB/s, so there was no
instruction-issue overhead left for wider loads/stores to remove.

**v3's ~13-19% regression was a real bug, not a real architectural
finding — corrected by v4.** `ncu` profiling (see the v3 section below)
found v3's kernel had a genuine coalescing bug: each thread was given a
contiguous *block* of 4 `float4`s, which means at any single unrolled
load instruction, consecutive threads in a warp land 64 bytes apart
instead of 16 — a classic stride-between-threads mistake, directly
confirmed by `ncu`'s own diagnostics ("only 16.0 of the 32 bytes
transmitted per sector are utilized... caused by a stride between
threads", 49.86% excessive sectors from uncoalesced access). The
original v3 write-up guessed a different cause (warp-parallelism
already saturated) — that guess was **wrong**, corrected here now that
real counter data exists; see the v3 section for what's left of it as a
"here's what NOT owning a coalescing bug costs you" data point.

v4 fixes it by striping the unroll across the whole grid instead of
giving each thread a private contiguous block (thread `g`'s k-th access
is `A[g + k*totalThreads]`, not `A[g*UNROLL+k]`) — every unrolled load
is now coalesced exactly like v2's. Result: **v4 measures at parity with
v1/v2** (~148 GB/s, no regression) at every VRAM-resident size tested so
far. That's expected on this card (still bandwidth-saturated, no
headroom for ILP to show a *win* here either) — the point of fixing it
wasn't to beat v1 locally, it was to remove a confound before this
technique gets tried on hardware where the ILP argument might actually
matter (see "On 'beating the leaderboard'" in `README.md`).

`solution.cu` stays on the v1 kernel — still the simplest version with
positive-or-neutral evidence at every size measured — but v4 is now the
one worth actually submitting to tensara alongside it, not v3.

## v1: naive, one thread per element — measured

| N | runtime | GB/s |
|---|---|---|
| 2^20  | 87.8 µs   | 143.30 |
| 2^22  | 341.1 µs  | 147.57 |
| 2^23  | 679.4 µs  | 148.17 |
| 2^25  | 2.7027 ms | 148.98 |
| 2^26  | 5.4060 ms | 148.97 |
| 2^27  | 11.0259 ms| 146.08 |
| 2^28  | 22.0292 ms| 146.23 |
| 2^29  | 191.235 ms| 33.69 (VRAM-oversubscribed, see below) |
| 2^30  | 472.883 ms| 27.25 (VRAM-oversubscribed, see below) |

`nsys` cross-check at N=2^23 (5 iterations, `cuda_gpu_kern_sum`): kernel
`vecAdd` averages **677.6 µs** per launch — matches `bench.exe`'s own
678.9 µs at the same size to within noise, confirming the `cudaEvent`
timing is measuring what it claims to.

**~148 GB/s is the practical ceiling on this card** for sizes that fit in
VRAM (2^20 through 2^28) — consistent with the ~147 GB/s figure noted in
`01-conv1d/benchmarks.md` for the same GTX 1650. Achieved bandwidth climbs
slightly with N up to 2^25 (fixed kernel-launch overhead amortizing over
more work) then holds flat — textbook memory-bandwidth-bound behavior.

## v2: float4 vectorized — measured

| N | runtime | GB/s | vs v1 |
|---|---|---|---|
| 2^20  | 89.4 µs   | 140.76 | −1.8% (noise) |
| 2^22  | 341.2 µs  | 147.53 | ~0% |
| 2^23  | 678.8 µs  | 148.29 | +0.1% |
| 2^25  | 2.7052 ms | 148.85 | −0.1% |
| 2^26  | 5.4157 ms | 148.70 | −0.2% |
| 2^27  | 11.0618 ms| 145.60 | −0.3% |
| 2^28  | 22.1194 ms| 145.63 | −0.4% |
| 2^29  | 187.138 ms| 34.43  | +2.2% (still within paging noise) |
| 2^30  | 473.725 ms| 27.20  | −0.2% (within paging noise) |

Confirms the README's prediction: cutting instruction/transaction count
4x has no effect when the kernel was already bandwidth-saturated, not
issue-rate-limited. This is a genuine (negative) result worth keeping —
same "document what didn't work and why" convention as `01-conv1d`'s v5.

## v3: float4 + 4x unrolled ILP — measured

| N | runtime | GB/s | vs v1 |
|---|---|---|---|
| 2^20  | 108.3 µs  | 116.14 | −18.9% |
| 2^22  | 414.1 µs  | 121.53 | −17.6% |
| 2^23  | 793.5 µs  | 126.86 | −14.4% |
| 2^25  | 3.2004 ms | 125.81 | −15.6% |
| 2^26  | 6.3689 ms | 126.44 | −15.1% |
| 2^27  | 12.6068 ms| 127.76 | −12.5% |
| 2^28  | 25.4336 ms| 126.65 | −13.4% |
| 2^29  | 900.703 ms| 7.15   | −78.8% (VRAM-oversubscribed, see below) |
| 2^30  | 2339.28 ms| 5.51   | −79.8% (VRAM-oversubscribed, see below) |

`ptxas -v` showed 38 registers/thread, 0 bytes spilled (vs. v1's 8
registers/thread) — real, but as suspected at the time, not the actual
cause (neither number is near Turing's 1024-thread/SM occupancy ceiling).

**Root cause, confirmed by `ncu --set full` (run with admin/elevated
counter access, N=2^25):**

| metric | value | reading |
|---|---|---|
| DRAM Throughput | 89.91% | looks fine in isolation, but... |
| Achieved Occupancy | 72.15% (vs. 100% theoretical) | some loss here too |
| Warp Cycles Per Issued Instruction | 518.4 cycles | very high for a memory-bound kernel |
| L1TEX Global Load pattern | **"only 16.0 of the 32 bytes transmitted per sector are utilized... could possibly be caused by a stride between threads"** (Est. Speedup: 25.23%) | — |
| L1TEX Global Store pattern | same finding, same estimate, for stores | — |
| Source Counters | **"uncoalesced global accesses resulting in 12,582,912 excessive sectors (50% of the total 25,165,824 sectors)"** (Est. Speedup: 49.86%) | — |

This is a genuine coalescing bug, not a latency/occupancy story: v3 gave
each thread a contiguous *block* of `UNROLL=4` float4s (`A[4g], A[4g+1],
A[4g+2], A[4g+3]` for thread `g`). Per-thread that looks like sequential
access, but at any single *unrolled* load instruction — which all 32
threads in a warp execute simultaneously — consecutive threads' addresses
are `4×16B = 64B` apart instead of the `16B` needed for full coalescing.
Classic block-interleaved-vs-striped mistake. Full report:
`ncu_summary_v3.txt` (generated via `ncu --import`, committed alongside
this file).

**Fixed in v4** (see below) by striping the unroll across the whole grid
instead of across a private per-thread block, restoring full coalescing
while keeping the actual thing being tested (multiple independent loads
in flight per thread before any store).

**The regression was far larger (~80%) once VRAM was oversubscribed**
(2^29/2^30) — plausibly the same coalescing problem compounding with
WDDM's page-fault-driven paging (more scattered per-sector touches means
more distinct pages touched per block), but this wasn't independently
confirmed and v4 should be re-measured at those sizes to see if fixing
coalescing also fixes this.

## v4: float4 + 4x striped ILP (coalescing bug fixed) — measured

| N | runtime | GB/s | vs v1 |
|---|---|---|---|
| 2^20  | 88.5 µs   | 142.19 | −0.8% (noise) |
| 2^22  | 342.2 µs  | 147.07 | −0.3% |
| 2^23  | 681.5 µs  | 147.70 | −0.3% |
| 2^25  | 2.7185 ms | 148.11 | −0.6% |
| 2^26  | 5.4420 ms | 147.98 | −0.7% |
| 2^27  | *pending* | | |
| 2^28  | *pending* | | |
| 2^29  | *pending* | | |
| 2^30  | *pending* | | |

Parity with v1/v2 at every size measured so far — the coalescing fix
recovered the full ~13-19% v3 lost. `ncu` confirmation of the fix itself
(that v4 no longer shows the "stride between threads" / excessive-sector
findings) is pending — see `PROFILE.md` for the command to re-check this.

This is the version worth actually submitting to tensara alongside v1/v2
if the ILP-hides-latency idea is going to be tested at all: it tests the
idea cleanly, without a coalescing bug muddying the result the way v3
did.

## VRAM oversubscription at N=2^29 / 2^30

This GTX 1650 has 4 GiB of VRAM. Three `N`-length `float` arrays at
N=2^29 need `3 × 2^29 × 4 B = 6 GiB`; at N=2^30, `12 GiB`. Both exceed
physical VRAM, but `cudaMalloc` doesn't fail — Windows WDDM transparently
backs the excess with system RAM and pages it over PCIe on demand. The
result is a cliff from ~148 GB/s down to **~27-34 GB/s**, in the
ballpark of PCIe 3.0 x16's ~16 GB/s one-direction / ~32 GB/s
bidirectional theoretical bandwidth, not the GPU's GDDR5 bandwidth at
all — a completely different bottleneck than what the kernel is being
graded on.

This is a local-hardware artifact, not a kernel correctness or
performance issue: tensara's actual grading GPUs (T4/L40S/etc.) have
16 GB+ of VRAM, comfortably fitting all three arrays at every test size
in this problem (`2^30` floats × 4 B × 3 = 12 GiB, fits in a 16 GB+ card
with room to spare) — so this ceiling should not appear on the real
grading hardware. Noted here only so a future local run isn't misread as
"the kernel got 5x slower."

## Block-size sweep (v1, N=2^25) — confirms launch config isn't the bottleneck

| threads/block | GB/s |
|---|---|
| 128  | 148.98 |
| 256  | 148.97 |
| 512  | 148.90 |
| 1024 | 147.51 |

Flat across 128-512, a hair worse at 1024 (fewer resident blocks per SM at
the max thread count leaves less latency-hiding headroom, though the
effect is tiny for a kernel this simple). 256 (the value `solution.cu`
uses) is already effectively optimal — there's no launch-configuration
tuning left to do for this kernel either, consistent with everything else
in this file pointing at "bandwidth-saturated, nothing left to trade."

## Target

Tensara's live leaderboard (checked 2026-08-15) tops out around
**82-86 µs on B200** for the current top ~15 entries — a card with
roughly **60x** this GTX 1650's memory bandwidth (~8 TB/s HBM3e vs.
~148 GB/s GDDR5). None of the local numbers above can be used to predict
standing on that hardware; a 60x bandwidth difference is a different
enough regime that conclusions don't transfer in either direction (see
`README.md`, "On 'beating the leaderboard'"). v1, v2, and v4 are all
correctness-checked and ready to submit (v3 is superseded by v4 — same
idea, coalescing bug fixed); only an actual tensara submission on their
B200 can say which one actually wins there.
