# Benchmarks — 02 Vector Addition

## Summary (GTX 1650)

> **Methodology note.** All numbers are plain-timed (`bench.exe N [iters]`,
> a 20-iteration averaged `cudaEvent` loop, 5 iterations at the two
> largest N to keep run time reasonable). GB/s = `3*N*4 bytes / time`
> (2 reads + 1 write, the minimum possible traffic for this problem — no
> version does any extra passes over the data).

| N | v1 (naive) | v2 (float4) | winner |
|---|---|---|---|
| 2^20 (1,048,576)   | 143.30 GB/s | 140.76 GB/s | tie (noise) |
| 2^22 (4,194,304)   | 147.57 GB/s | 147.53 GB/s | tie |
| 2^23 (8,388,608)   | 148.17 GB/s | 148.29 GB/s | tie |
| 2^25 (33,554,432)  | 148.98 GB/s | 148.85 GB/s | tie |
| 2^26 (67,108,864)  | 148.97 GB/s | 148.70 GB/s | tie |
| 2^27 (134,217,728) | 146.08 GB/s | 145.60 GB/s | tie |
| 2^28 (268,435,456) | 146.23 GB/s | 145.63 GB/s | tie |
| 2^29 (536,870,912) | 33.69 GB/s  | 34.43 GB/s  | tie (both PCIe-paging-bound, see below) |
| 2^30 (1,073,741,824) | 27.25 GB/s | 27.20 GB/s | tie (both PCIe-paging-bound, see below) |

**v2 (float4 vectorization) does not beat v1 at any measured size** — every
difference is within normal run-to-run noise (≤2%). See `README.md` for
why: v1 was already bandwidth-saturated at ~148 GB/s, so there was no
instruction-issue overhead left for wider loads/stores to remove.
`solution.cu` stays on the v1 kernel (simpler code, identical performance).

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

Tensara's leaderboard runs on data-center GPUs (T4/L40S/H100/B200) with
far higher HBM/GDDR6 bandwidth than this GTX 1650's GDDR5. Since v1/v2
are already at this card's bandwidth ceiling with no algorithmic
improvement available (see `README.md`), the expected result on tensara
is the same kernel scaling with that card's bandwidth — not a case where
further local kernel changes would help.
