# 01 — 1D Convolution

[tensara.org/problems/conv-1d](https://tensara.org/problems/conv-1d) · Difficulty: Easy (rated) / bandwidth-heavy in practice at the given sizes

## Problem

```
C[i] = sum_{j=0..K-1} A[i + j - r] * B[j],   r = (K-1)/2
```

Zero padding outside `A`, `K` odd, output size `N` (same as input). This is
exactly `torch.nn.functional.conv1d(A, B, padding=K//2)` (cross-correlation,
kernel not flipped).

Test sizes: `N ∈ {65536, 32768, 131072, 524288}`, `K = 8191` for all of them.

## Why this isn't a "just write the triple loop" problem

`K = 8191` is *enormous* relative to typical conv kernels (usually 3–11).
With `N = 524288`, the raw compute is:

```
N * K ≈ 4.29 billion multiply-adds  (≈ 8.6 GFLOP)
```

On a T4 (≈8.1 TFLOP/s FP32), that's under 2ms of pure compute — so this
problem is **not** compute-bound. It's bound by how many times each element
of `A` gets re-read from memory. A naive kernel where every thread
independently walks `A[i-r .. i-r+K-1]` from global memory re-reads most of
`A` up to `K` times total (once per nearby output), and does it through
whatever cache happens to be warm — no guarantee of reuse.

The two things that actually determine performance here:

1. **How fast can `B` (the kernel) be re-read by every thread, every
   iteration, without wasting bandwidth?** → constant memory.
2. **How much of the redundant re-reading of `A` can be turned into on-chip
   reuse instead of repeated DRAM/L2 traffic?** → shared-memory tiling.

## v1: shared-memory tiled, O(N·K) (`solution_tiled.cu`)

Superseded by the FFT version below as the actual `solution.cu` submitted to
tensara — kept here (and in the repo) because the reasoning still applies
whenever K is small enough that FFT setup overhead isn't worth paying.

### 1. `B` lives in `__constant__` memory

All 8191 kernel weights (32KB) fit comfortably in CUDA's 64KB constant
memory. Every thread in a warp executes the same loop iteration `j` at the
same time (the inner loop is a plain `for`, no divergence), so every thread
in the warp requests `B[j]` simultaneously — the textbook case for constant
memory's broadcast path, which serves the whole warp from a single cache
line fetch. This is strictly better than a global-memory read through
`__ldg`, which still works (broadcasts fine through L1) but constant memory
gives a dedicated cache with guaranteed broadcast semantics regardless of
what else is contending for L1.

### 2. `A` is staged into shared memory per block, with halo

Each block owns `TILE` consecutive output elements. To compute all of them
it needs `A[blockStart - r .. blockStart + TILE - 1 + r]` — `TILE + K - 1`
elements. The block cooperatively loads exactly that range into shared
memory once (`__syncthreads()`), and every thread then does its whole
`K`-length dot product out of shared memory instead of global memory.

This means each element of `A` is read from **global memory only once per
block whose halo covers it**, instead of once per output element that needs
it. Concretely, each global element of `A` is re-read from DRAM/L2
`(TILE + K - 1) / TILE` times across all blocks (≈9x with `TILE=1024,
K=8191`) instead of up to `K` times (8191x) with no staging at all — and
even that residual redundancy mostly hits L2 rather than DRAM, since `A`
for the largest test case is only 2MB and T4's L2 is 4MB.

`TILE` is computed at launch time from the device's actual max shared
memory per block (`cudaDevAttrMaxSharedMemoryPerBlockOptin`), so it adapts
to the GPU it's run on and to whatever `K` a future test case uses, instead
of being hardcoded for T4 + K=8191.

### 3. Fallback path

If `K` is so large that not even a 32-wide tile's halo fits in the device's
max shared memory (not the case for any of this problem's test cases), the
code falls back to a direct kernel that skips shared-memory tiling but
still uses constant memory for `B` and `__ldg` for `A`, relying on L2
residency for reuse. This keeps the solution correct (not just fast) across
input shapes well outside what's tested here.

## Complexity

- Global memory traffic for `A`: `O(N * (TILE+K-1)/TILE)` ≈ `O(N)` for
  `TILE >> 1` (dominated by DRAM only on first touch; steady-state reuse
  comes from L2 for the ~9x redundant halo reads).
- Global memory traffic for `B`: `O(K)` once (the `cudaMemcpyToSymbol`),
  effectively free thereafter (constant cache).
- Compute: `O(N*K)`, ~8.6 GFLOP for the largest case, not the bottleneck.
- Shared memory per block: `(TILE + K - 1) * 4` bytes, chosen to fit under
  the device's opt-in max (typically 64–228KB depending on architecture).

## v2: FFT-based (this is what's in `solution.cu` now)

The tiled kernel above is a real optimization over naive direct convolution,
but it's still fundamentally O(N·K) — and for K=8191 that's a hard floor of
low-single-digit milliseconds no amount of shared-memory tiling gets you
under, on any GPU. Leaderboard entries on tensara run in **microseconds**,
which is only possible with an asymptotically different algorithm:
convolution via FFT, O((N+K) log(N+K)) instead of O(N·K) — for the largest
test case that's roughly a 400x reduction in operation count, not a
constant-factor speedup.

**The trick**: cross-correlation (what this problem asks for) is a
convolution with the kernel reversed. So:

1. Zero-pad `A` with `r` zeros on each side → `Apad`, length `N+K-1`.
2. Reverse `B` → `Brev` (length `K`).
3. Zero-pad both out to a common FFT size `L ≥ N+2K-2` (the full linear
   convolution length of `Apad` and `Brev`).
4. `R2C` FFT both, complex pointwise multiply, `C2R` FFT back (real-to-
   complex/complex-to-real transforms instead of full complex-to-complex,
   since the input is real — half the memory and roughly half the work).
5. `C[i]` is the length-`N` window of that result starting at offset `K-1`,
   divided by `L` (cuFFT's inverse transform doesn't auto-normalize).

`L` is chosen as the smallest 2/3/5/7-smooth size ≥ the minimum required
length — cuFFT's mixed-radix kernels are fast for smooth sizes and slow for
anything with a large prime factor, so a plain "round up to power of 2"
would waste up to ~2x the FFT length (and therefore real time) for no
reason.

**cuFFT plans and device buffers are cached across calls**, keyed by `L`.
Creating a cuFFT plan is not free — paying that cost inside every timed
call would be self-defeating when the whole point is microsecond-scale
runtimes. This assumes `solution()` gets called repeatedly for the same
input size within a process (true of tensara's benchmarking harness and of
`tests/bench.cu`, which both time an averaged loop, not a single cold call).

Correctness for this version was checked differently than the tiled kernel:
the index arithmetic (the reverse + pad + offset-slice scheme above) is
easy to get subtly wrong, so it's validated in `tests/validate_fft.py`
against numpy's FFT and a brute-force reference before being ported to
CUDA/cuFFT, rather than hand-verified in C++.

### Numerical note

FFT convolution in `float32` isn't bit-exact with direct summation — expect
relative error on the order of `1e-6` to `1e-5` for these sizes (verified
against a double-precision brute-force reference), which should be well
within any reasonable checker tolerance but is worth knowing about if a
comparison ever looks "off by a hair."

### `solution_tiled.cu`

The original shared-memory-tiled O(N·K) version is kept in this folder as
`solution_tiled.cu` — still a legitimate, well-reasoned kernel (see the
sections above), just not competitive against FFT-based entries at this
K. Useful as a baseline / for problems where K is small enough that FFT
overhead isn't worth it.

## Files

- `solution.cu` — the FFT-based version, paste this into tensara's editor.
- `solution_tiled.cu` — the earlier O(N·K) shared-memory-tiled version, kept
  for reference/comparison (see above).
- `tests/validate.cpp` — host-only (no GPU) correctness check of
  `solution_tiled.cu`'s tiling/halo index arithmetic, against a brute-force
  reference.
- `tests/validate_fft.py` — validates `solution.cu`'s FFT indexing scheme
  (reverse/pad/offset-slice) against numpy and a brute-force reference.
- `tests/bench.cu` — GPU benchmark driver, calls whichever `solution()` is
  currently linked in.
- `benchmarks.md` — real measured results.

## Known assumption to double check

The tensara editor's starter template signature is
`extern "C" void solution(const float* A, const float* B, float* C, /* N, K */)`
but the exact integer type for `N`/`K` (`size_t` vs `int`) isn't visible in
the problem statement. This solution assumes `size_t`, matching the
convention used by tensara's other problems (vector-add, matmul, etc.). If
tensara's actual harness declares the last two parameters as `int` instead,
the fix is a one-line signature change — everything else is unaffected
since `N` and `K` are cast to `int` internally anyway (well within `int`
range for these sizes).
