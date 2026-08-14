Profiling guide for conv1d
=========================

Build
-----

Compile with line information so Nsight Compute can correlate source:

```bash
nvcc -O3 -arch=sm_80 -lineinfo -o conv1d.exe solution.cu tests/validate.cpp
```

Quick (full) profile
--------------------

Run a full collection (slower, comprehensive):

```bash
ncu --target-processes all --set full -o conv1d_report conv1d.exe
```

Metric-subset (faster)
----------------------

Collect a focused metric set to iterate quickly:

```bash
ncu --metrics achieved_occupancy,sm__cycles_elapsed.avg,sm__ipc.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed -o conv1d_metrics conv1d.exe
```

Profile a specific kernel name (filtering)
-----------------------------------------

Limit profiling to kernels whose names match a regex (e.g. `conv1d_tiled`):

```bash
ncu --kernel-name "^conv1d_tiled" -o conv1d_tiled_only conv1d.exe
```

Launch-based/sampling options
----------------------------

- Use `--launch-count <N>` to profile only the first N launches.
- Add `--target-processes all` to include child processes if your test spawns any.

Open and inspect
-----------------

- The run writes `conv1d_report.ncu-rep` (or similarly named). Open it with the GUI:

```bash
ncu-ui conv1d_report.ncu-rep
```

What to look at
----------------

- Summary: dominant kernel(s) by elapsed time.
- Source view / source correlation: inspect the inner loop (the j-loop).
- Achieved occupancy: low values indicate shared/reg pressure or large dynamic shared mem.
- SM IPC and cycles: low IPC with high memory traffic suggests memory bound.
- DRAM / L2 / Cache counters: inspect load/store throughput to find bandwidth limits.

Quick tuning checklist
---------------------

- If occupancy is low: reduce `TILE` (dynamic shared mem) or reduce register usage.
- If memory bound: favor the tiled variant so inputs are reused from shared memory.
- Compare `useConst` vs non-const: constant memory reads can show broadcast benefits.

Windows / environment notes
---------------------------

- On Windows use the `ncu` binary from Nsight Compute. If `ncu` is not on PATH, run it from its install folder.
- Consider running your exe under administrator if permission issues arise when accessing GPU counters.

Example fast iteration cycle
----------------------------

1. Build with `-lineinfo`.
2. Run `ncu --metrics achieved_occupancy,sm__ipc.avg,dram__throughput.avg -o quick conv1d.exe`.
3. Open `ncu-ui quick.ncu-rep`, inspect kernel hotspots and memory metrics.
4. Change `TILE` or compile-time options, rebuild, re-run.

Further reading
---------------

- NVIDIA Nsight Compute documentation: https://developer.nvidia.com/nsight-compute
- Look up specific metric names (prefix `sm__`, `dram__`, `l1tex__`) in the Nsight docs for interpretation.
