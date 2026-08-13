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

## Approach

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

## Files

- `solution.cu` — paste this directly into tensara's CUDA C++ editor.
- `tests/validate.cpp` — host-only (no GPU) correctness check of the exact
  tiling/halo index arithmetic, against a brute-force reference. See root
  README for how to run it.
- `benchmarks.md` — real submission results (fill in after running on
  tensara — see that file for the template).

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
