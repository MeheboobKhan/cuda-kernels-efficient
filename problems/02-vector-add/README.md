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

## v3: float4 + 4x unrolled ILP (`solution_v3.cu`) — locally worse, unverified on real target hardware

Prompted by a look at tensara's live leaderboard: top entries run **~82-86 µs
on B200**, a card with roughly **60x** this local GTX 1650's memory
bandwidth (~8 TB/s HBM3e vs. ~148 GB/s GDDR5). That gap matters for which
optimizations are worth trying:

- On this GTX 1650, v1's scalar accesses already coalesce into full
  128-byte transactions and saturate the bus — there is nothing left for
  a wider-transaction kernel to fix, which is exactly what v2 showed.
- On a part with ~60x the bandwidth, Little's Law says the
  bytes-in-flight needed to keep the pipe full scales up by roughly the
  same factor. If per-SM occupancy doesn't naturally supply enough
  independent outstanding requests, a kernel can under-fill a
  high-bandwidth bus even while "coalesced" — the standard fix is more
  memory-level parallelism per thread (issue several independent loads
  before any dependent store), on top of wide transactions.

v3 does both: each thread issues 4 independent `float4` loads from `A`
and 4 from `B` (16 floats total) before writing anything back, instead of
one load/store pair at a time.

**Measured result: v3 is *worse* than v1/v2 on the GTX 1650** — ~126 GB/s
vs. ~148 GB/s, a ~15% regression, consistent across every VRAM-resident
size tested. `ptxas -v` shows 38 registers/thread and **zero spilling**,
so it isn't the obvious register-pressure explanation; the more likely
cause is that this card's SM already has enough resident warps to hide
memory latency through ordinary warp-level interleaving, and batching 8
loads before any store per thread creates a stall bubble that a card
already saturated at the warp-interleaving level doesn't benefit from —
without `ncu` (blocked here, see `PROFILE.md`) this is a reasoned
hypothesis, not a confirmed root cause.

**This result cannot be used to reject v3 for the actual target
(tensara's grading GPU).** The mechanism v3 targets — hiding memory
latency behind more in-flight requests — specifically matters *more* the
higher the bandwidth, i.e. it should matter least on this card and most
on a B200. A GTX-1650-measured regression is real information about the
GTX 1650, not proof about a B200. The only way to know whether v3 beats
v2 on the hardware that actually determines leaderboard rank is to
submit both and compare — see the "beating the leaderboard" note below.

`solution.cu` stays on v1: it's the only version with *measured* evidence
of being at-or-above every alternative, on the only hardware available
to measure on.

## On "beating the leaderboard"

Tensara's leaderboard runs on GPUs (B200 for the current top entries)
that aren't available locally. Nothing measured here — including v3's
local regression — can be used to predict standing on that hardware;
~60x more bandwidth is a different enough regime that conclusions don't
transfer either direction. Getting a real answer requires an actual
submission through tensara's own grader. v1, v2, and v3 are all here,
correctness-checked and ready to paste, specifically so that question can
be answered empirically rather than guessed at locally.

## Files

- `solution.cu` — currently mirrors `solution_v1.cu` (the naive kernel),
  since v2 didn't win. Paste this into tensara's editor.
- `solution_v1.cu` — naive one-thread-per-element baseline.
- `solution_v2.cu` — `float4`-vectorized variant, kept for reference even
  though it didn't measurably improve on v1 (see above).
- `solution_v3.cu` — `float4` + 4x-unrolled ILP variant, measured *worse*
  locally but built for a higher-bandwidth target than this card (see
  above) — a candidate worth actually submitting to tensara, not a
  rejected idea.
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
