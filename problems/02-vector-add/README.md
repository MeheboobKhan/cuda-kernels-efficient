# 02 — Vector Addition

[tensara.org/problems/vector-addition](https://tensara.org/problems/vector-addition) · Difficulty: Easy

## Problem

```
C[i] = A[i] + B[i]
```

Elementwise addition of two vectors `A`, `B` of length `N`, producing `C` of
length `N`.

Test sizes: `N ∈ {2^20, 2^22, 2^23, 2^25, 2^26, 2^29, 2^30}`.

## Why there isn't much to optimize here

Vector add does **1 FLOP per 12 bytes of traffic** (read `A[i]`, read
`B[i]`, write `C[i]`, one add). On any GPU built in the last decade, compute
throughput vastly outpaces memory bandwidth at that ratio - this kernel is
memory-bandwidth-bound as completely as a kernel can be, with essentially
zero compute to hide behind. Unlike `01-conv1d` (compute-bound at large K,
with real algorithmic wins available via FFT), there is no smarter
algorithm for vector add - the only question is how close a given kernel
gets to the card's physical memory bandwidth ceiling.

## v1: naive, one thread per element (`solution_v1.cu`)

```cuda
if (i < n) C[i] = A[i] + B[i];
```

That's it. Global reads of `A[i]` and `B[i]` are already fully coalesced
(consecutive threads touch consecutive addresses), so there's no tiling,
shared memory, or constant memory to reach for - none of those help a
kernel that reads each byte exactly once and does no reuse.

**Measured on a GTX 1650: ~148 GB/s**, right at the card's bandwidth
ceiling (see `benchmarks.md`). The naive kernel is already
bandwidth-saturated.

## v2: vectorized `float4` loads/stores (`solution_v2.cu`)

The standard next lever when a kernel is bandwidth-bound but still shows
scheduling/instruction-issue overhead: issue one 16-byte `float4`
load/store per thread instead of four separate 4-byte loads/stores. This
cuts the instruction count and memory-transaction count 4x for the same
amount of data, which sometimes recovers bandwidth a scalar kernel leaves
on the table due to per-thread overhead.

`n % 4` isn't always 0 for arbitrary input, so a scalar tail kernel
handles the remainder - the vectorized main pass and the tail kernel
together cover all `n`, not just multiples of 4. Every one of this
problem's actual test sizes (`2^20` and up) is exactly divisible by 4,
so the tail path never fires on the real test cases, but it's there for
correctness on arbitrary `n` (validated in `tests/bench.cu --check`).

**Result: no measurable improvement over v1** (see `benchmarks.md`) -
within run-to-run noise at every tested size. This is expected, not a
bug: v1 was *already* at the hardware bandwidth ceiling, so there was no
instruction-issue overhead left for vectorization to remove. Vectorized
loads help when a kernel is leaving bandwidth on the table due to
per-thread overhead; they don't help a kernel that's already saturating
the memory bus. Documented here as a real (negative) result, not
dropped, per the same "keep what you tried" convention as `01-conv1d`.

## Files

- `solution.cu` — currently mirrors `solution_v1.cu` (the naive kernel),
  since v2 didn't win. Paste this into tensara's editor.
- `solution_v1.cu` — naive one-thread-per-element baseline.
- `solution_v2.cu` — `float4`-vectorized variant, kept for reference even
  though it didn't measurably improve on v1 (see above).
- `tests/bench.cu` — GPU benchmark + on-device correctness harness.
- `benchmarks.md` — measured results.
- `PROFILE.md` — profiling guide (`ncu`/`nsys`) for this problem.

## Local hardware ceiling

On the local GTX 1650 (~147-149 GB/s achievable, per `01-conv1d`'s
measurements and confirmed again here), vector add tops out around
**148 GB/s**. Tensara's leaderboard runs on data-center GPUs (T4/L40S/etc.)
with substantially higher HBM/GDDR6 bandwidth, so a submission there
should scale with that card's bandwidth, not this one's - the kernel
itself doesn't need to change for a faster card, since it's already
bandwidth-optimal.
