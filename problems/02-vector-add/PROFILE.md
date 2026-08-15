Profiling guide for vector-add
===============================

Build
-----

Compile with line information so Nsight tools can correlate source. On
Windows/Git-Bash, nvcc needs to be pointed at the 64-bit host compiler
explicitly or it may pick a 32-bit `cl.exe` off PATH and fail with bogus
`size_t` redeclaration errors:

```bash
CC="C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.28.29333/bin/Hostx64/x64/cl.exe"
nvcc -O3 -arch=sm_75 -lineinfo -ccbin "$CC" -o bench.exe solution.cu tests/bench.cu
```

`ncu` (Nsight Compute) — hardware counters
-------------------------------------------

Full profile:

```bash
ncu --set full --launch-count 1 --launch-skip 1 -o vecadd_report -f ./bench.exe <N> 1
```

(`--launch-skip 1` skips the warmup launch inside `solution()`'s first
call so the profiled launch is the timed one, same convention as
`01-conv1d`.)

**Known local limitation (session-dependent):** `ncu` requires GPU
performance-counter access that Windows restricts to Administrator by
default (`NVreg_RestrictProfilingToAdminUsers`). A non-admin shell fails
with:

```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

This blocked `ncu` in the Git Bash session used for most of this
problem's benchmarking, but **worked fine from an elevated "x64 Native
Tools Command Prompt"** on the same machine — the restriction is
per-session/per-privilege-level, not a hard block on the hardware. If
`ncu` fails in one shell, try again from an Administrator-elevated one
before assuming it's unavailable. `nsys` (below) needs no elevation
either way and is the fallback when elevation isn't an option.

**Windows/cmd.exe syntax note:** in a plain `cmd.exe` prompt, drop the
`./` (`bench.exe` or `.\bench.exe` both work, `./bench.exe` does not),
use `tests\bench.cu` (backslash) or `tests/bench.cu` (nvcc accepts
either), and `for %n in (a b c) do cmd %n` instead of bash's
`for n in a b c; do cmd $n; done`. Env vars are `set VAR=value` then
`%VAR%`, not `VAR=value` inline or `$VAR`.

**Reading a saved report without re-profiling:** `ncu -o <name> -f`
writes a binary `.ncu-rep` but (in this `ncu` version) doesn't print the
metrics table to the console. Get the text version with:

```
ncu --import <name>.ncu-rep > <name>_summary.txt
```

This is how `ncu_summary_v3.txt` and `ncu_summary_v4.txt` in this
directory were produced — same convention as `01-conv1d/ncu_summary.txt`.

`nsys` (Nsight Systems) — timeline / kernel duration, no admin needed
----------------------------------------------------------------------

Activity-based tracing (kernel start/stop timestamps, memcpy timing) does
**not** hit the same permission wall as `ncu`'s hardware counters, so this
works in a normal user session:

```bash
nsys profile --force-overwrite=true -o vecadd_nsys --stats=true ./bench.exe <N> <iters>
```

Useful sections in the printed stats: `cuda_gpu_kern_sum` (per-kernel
average duration — this is what `bench.exe`'s own GB/s number is derived
from, cross-check the two), `cuda_gpu_mem_time_sum` (H2D/D2H copy time,
irrelevant to the timed loop but useful for sanity-checking total
`bench.exe --check` runtime).

What to look at
----------------

- For this problem, the main question is **achieved GB/s vs. the card's
  bandwidth ceiling** — there's no tiling/shared-memory/cache-reuse story
  for a pure elementwise kernel (see `README.md` for why). `bench.exe`'s
  own `%.2f GB/s` output (computed as `3*N*4 bytes / time`) is normally
  enough for that; `ncu`'s `DRAM Throughput %` is the same information
  from the hardware counters, useful as a cross-check.
- **Coalescing is still worth checking even here**, though — it's not
  automatic just because the kernel is "simple." v3's ~15% regression
  (see `benchmarks.md`) turned out to be a real coalescing bug in a
  hand-unrolled kernel, caught by `ncu`'s Memory Workload Analysis Tables
  section ("only 16.0 of the 32 bytes transmitted per sector are
  utilized... caused by a stride between threads") and Source Counters
  ("uncoalesced global accesses, N excessive sectors"). Any time a kernel
  gives one thread multiple elements to handle, check this section before
  trusting the access pattern is still coalesced — "looks sequential
  per-thread" is not the same as "coalesced per-warp-instruction."
- If GB/s is well below the card's rated bandwidth at large N, suspect
  launch-configuration overhead (too few blocks to saturate all SMs'
  memory pipes) rather than anything algorithmic — there's no algorithm
  to improve for a pure elementwise kernel.
- Watch for VRAM oversubscription at the largest test sizes (`2^29`,
  `2^30`): if the three arrays don't fit in the card's physical memory,
  Windows WDDM transparently pages over PCIe instead of failing, and GB/s
  drops off a cliff (measured ~25-35 GB/s vs. ~148 GB/s in-VRAM on a 4GB
  GTX 1650 — see `benchmarks.md`). This looks like a regression in the
  kernel if you don't know to check `nvidia-smi` memory usage first; it
  isn't one.

Windows / environment notes
---------------------------

- `ncu` ships as `ncu.bat`; if not on PATH, find it under
  `C:\Program Files\NVIDIA Corporation\Nsight Compute <version>\`.
- `nsys` ships under
  `C:\Program Files\NVIDIA Corporation\Nsight Systems <version>\target-windows-x64\nsys.exe`.
- Both need the 64-bit `cl.exe` fix above at build time, independent of
  which profiler is used at run time.
