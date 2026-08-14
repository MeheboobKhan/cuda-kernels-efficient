# Benchmarks — 01 Conv1D

Measured with `tests/bench.cu` (20 iterations, averaged, warmup call excluded)
on a local **GTX 1650** (Turing, sm_75, ~2.85 TFLOP/s FP32 peak, ~128 GB/s
memory bandwidth). tensara's official runs happen on a **T4** (~8.1 TFLOP/s
FP32, ~320 GB/s bandwidth), so expect these to undersell the actual
leaderboard result — submit `solution.cu` at
https://tensara.org/problems/conv-1d for the number that counts.

| N | K | GPU | Runtime | GFLOP/s |
|---|---|-----|---------|---------|
| 65536  | 8191 | GTX 1650 | 2.7396 ms | 391.9 |
| 32768  | 8191 | GTX 1650 | 1.3535 ms | 396.6 |
| 131072 | 8191 | GTX 1650 | 5.4218 ms | 396.0 |
| 524288 | 8191 | GTX 1650 | 16.1058 ms | 533.3 |

## Reading these numbers

Throughput sits at a near-constant ~392-400 GFLOP/s for the three smaller
sizes, then jumps to 533 GFLOP/s at the largest N. That's not noise — it's
the `cudaMemcpyToSymbol` call in `solution()` that re-uploads the 32KB
kernel to constant memory on *every* invocation (needed because the
problem's function signature has no persistent state between calls). That
copy is a fixed ~K-sized cost, so it eats a bigger fraction of the total
runtime when there's less compute to amortize it over. At N=524288 the
actual convolution work dominates enough that we get closer to the kernel's
real steady-state throughput.

On a T4 in tensara's actual test harness, this fixed cost is comparatively
smaller (higher memory bandwidth moves those 32KB faster), so expect the
gap between small-N and large-N GFLOP/s to be less pronounced there.

None of these are anywhere near the GTX 1650's ~2.85 TFLOP/s peak, which is
expected and by design — this kernel targets memory/reuse-bound behavior
(see the problem README's roofline discussion), not raw FLOP throughput.

## Reproducing

```powershell
nvcc -O3 -arch=sm_75 -lineinfo -o bench.exe solution.cu tests\bench.cu
.\bench.exe 524288 8191
```

Swap `-arch=sm_75` for whatever matches your GPU, and pass different `N K`
arguments to `bench.exe` (defaults to 524288 8191 if omitted).