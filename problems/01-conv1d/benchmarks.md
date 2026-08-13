# Benchmarks — 01 Conv1D

Not yet run on real hardware (this repo was authored in a sandbox without a
GPU). Submit `solution.cu` at https://tensara.org/problems/conv-1d and fill
in the table below with the numbers tensara reports.

| N | K | GPU | Runtime | GFLOP/s | Notes |
|---|---|-----|---------|---------|-------|
| 65536  | 8191 | T4 | — | — | |
| 32768  | 8191 | T4 | — | — | |
| 131072 | 8191 | T4 | — | — | |
| 524288 | 8191 | T4 | — | — | |

Theoretical floor for the largest case (N=524288): ~8.6 GFLOP of compute,
so on a T4 (~8.1 TFLOP/s FP32) compute alone would take ~1ms — actual
runtime will be higher due to being memory/reuse bound (see problem
README), which is exactly what the shared-memory tiling in `solution.cu`
targets.
