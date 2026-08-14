# Benchmarks — 01 Conv1D

## v2: FFT-based (`solution.cu`) — current

Not yet measured locally. Compile with `-lcufft` and run `bench.exe`:

```powershell
nvcc -O3 -arch=sm_75 -lineinfo -o bench.exe solution.cu tests\bench.cu -lcufft
.\bench.exe 524288 8191
.\bench.exe 65536 8191
.\bench.exe 32768 8191
.\bench.exe 131072 8191
```

| N | K | GPU | Runtime | Notes |
|---|---|-----|---------|-------|
| 65536  | 8191 | — | — | |
| 32768  | 8191 | — | — | |
| 131072 | 8191 | — | — | |
| 524288 | 8191 | — | — | |

Also worth grabbing the *first-call* (cold, plan-creation-included) time
separately from the steady-state loop average `bench.cu` reports — the
cache-across-calls design means those two numbers can differ a lot, and
which one matters depends on how tensara's harness actually times
submissions (single call vs. averaged loop).

For reference, tensara leaderboard entries for this problem currently run
in the **17–52 microsecond** range on data-center GPUs (L40S/B200/H100/T4)
— that's the target to compare against once a real submission goes in.

## v1: shared-memory tiled, O(N·K) (`solution_tiled.cu`) — superseded

Measured with the same `tests/bench.cu` harness on a local **GTX 1650**
(Turing, sm_75). Kept here as a reference point for how much the FFT
rewrite should be expected to win by — these are millisecond-scale, the
FFT version needs to land in microseconds to be competitive.

| N | K | GPU | Runtime | GFLOP/s |
|---|---|-----|---------|---------|
| 65536  | 8191 | GTX 1650 | 2.7396 ms | 391.9 |
| 32768  | 8191 | GTX 1650 | 1.3535 ms | 396.6 |
| 131072 | 8191 | GTX 1650 | 5.4218 ms | 396.0 |
| 524288 | 8191 | GTX 1650 | 16.1058 ms | 533.3 |

(GFLOP/s isn't a meaningful metric for the FFT version — it does ~400x
fewer FLOPs for the same output, so a direct GFLOP/s comparison between v1
and v2 would be misleading. Wall-clock runtime is the only fair comparison.)
