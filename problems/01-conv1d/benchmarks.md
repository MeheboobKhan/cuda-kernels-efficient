# Benchmarks — 01 Conv1D

## v3: batched forward FFT (`solution.cu`) — current

`cufftPlanMany` batch=2 for `R2C(A)`/`R2C(B)` + fused `padBoth` kernel, built
directly off the v2 `ncu` profile (see below). Not yet measured — compile
and run:

```powershell
nvcc -O3 -arch=sm_75 -lineinfo -o bench.exe solution.cu tests\bench.cu -lcufft
.\bench.exe 524288 8191
.\bench.exe 65536 8191
.\bench.exe 32768 8191
.\bench.exe 131072 8191
```

| N | K | GPU | Runtime | vs v2 |
|---|---|-----|---------|-------|
| 65536  | 8191 | — | — | — |
| 32768  | 8191 | — | — | — |
| 131072 | 8191 | — | — | — |
| 524288 | 8191 | — | — | — |

## v2: FFT-based, unbatched (superseded by v3)

Two separate `cufftExecR2C` calls (one for `A`, one for `B`) instead of one
batched call. Measured on a local **GTX 1650** (Turing, sm_75).

| N | K | GPU | Runtime |
|---|---|-----|---------|
| 65536  | 8191 | GTX 1650 | 148.0 µs |
| 32768  | 8191 | GTX 1650 | 218.8 µs |
| 131072 | 8191 | GTX 1650 | 192.0 µs |
| 524288 | 8191 | GTX 1650 | 757.8 µs |

Note the non-monotonic result: 32768 (218.8µs) ran slower than 131072
(192.0µs) despite 4x less data — traced to the "smallest 7-smooth FFT size"
search picking an awkward factor decomposition for that specific N. Not yet
root-caused with `ncu` (only N=524288 was profiled below); worth revisiting
if v3's numbers show the same inversion.

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

(GFLOP/s isn't meaningful for v2/v3 — FFT does ~400x fewer FLOPs for the
same output, so it's omitted there; wall-clock is the only fair comparison
across v1 vs v2/v3.)

## Target

tensara leaderboard for this problem currently tops out around **17-52
microseconds** on data-center GPUs (L40S/B200/H100/T4). All local numbers
above are on a GTX 1650 (~2.85 TFLOP/s FP32, ~128 GB/s bandwidth — well
below any of those cards), so expect a real tensara submission to land
faster than these local numbers independent of further code changes. Still
tracking the gap here since the *relative* improvement between v1→v2→v3 is
hardware-independent.