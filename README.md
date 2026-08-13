# cuda-hard-problems

Fast, carefully-optimized CUDA solutions to the "hard" problems on
[tensara.org](https://tensara.org) — with the reasoning behind every
optimization written down, not just the code.

Every problem folder follows the same layout so solutions stay comparable
and honest about what's actually been verified:

```
problems/<NN-slug>/
  solution.cu       # the file you paste into tensara's editor and submit
  README.md         # complexity analysis + why the kernel is built this way
  tests/            # CPU-only correctness harness (no GPU required to run)
  benchmarks.md      # measured runtimes from real tensara submissions (filled in after submitting)
```

## Philosophy

1. **Correctness first.** Every kernel's indexing logic is validated against
   a brute-force reference in `tests/`, using a host-only C++ harness that
   mirrors the kernel's exact index arithmetic. This catches the off-by-one
   errors that are the #1 source of wrong-answer submissions in convolution/
   stencil-style kernels, without needing a GPU in CI.
2. **Optimize for the actual bottleneck.** Most of these problems are memory-
   bandwidth- or reuse-bound, not compute-bound. Each README explains the
   roofline reasoning: what's the theoretical minimum data movement, and how
   close does the kernel get.
3. **No mystery meat.** No copy-pasted kernels without explanation. If a
   technique (shared-memory tiling, constant memory broadcast, warp shuffle
   reduction, tensor cores, etc.) is used, the README says why it applies
   *to this specific problem's shape*, since the "best" kernel for a
   convolution changes a lot depending on how big K is relative to N.

## Problems

| # | Problem | Technique | Status |
|---|---------|-----------|--------|
| 01 | [1D Convolution](problems/01-conv1d/) | constant-memory kernel + shared-memory input tiling with halo | ✅ solved, CPU-validated |

## Development environment note

This repo was authored in a sandbox without an NVIDIA GPU or the CUDA
toolkit available, so `solution.cu` files have **not** been compiled with
`nvcc` or run on real hardware here. Each problem's correctness is instead
verified by a host-only C++ harness (`tests/validate.cpp`) that reproduces
the kernel's exact index math on the CPU and checks it against a brute-force
reference. Before trusting a submission blindly, run the tests locally
(`g++` only, no CUDA needed) and/or submit directly to tensara, which
compiles and runs on real GPUs (T4 by default) and reports pass/fail plus
timing.

```bash
cd problems/01-conv1d/tests
g++ -O2 -std=c++17 -Wall -o validate validate.cpp && ./validate
```

## Contributing more problems

Copy `problems/01-conv1d/` as a template: write the kernel, write a CPU
emulator of its indexing scheme in `tests/validate.cpp`, get it green, then
fill in `benchmarks.md` with real numbers from a tensara submission.
