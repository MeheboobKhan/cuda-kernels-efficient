#include <cuda_runtime.h>

// v4: float4 + striped (grid-stride-style) ILP unroll - fixes v3's coalescing bug.
//
// ncu profiling of v3 (see benchmarks.md) found the real cause of its
// regression, and it was NOT the "already warp-parallelism-saturated"
// hypothesis originally guessed in v3's comments: it was a genuine
// coalescing bug. v3 gave each thread a contiguous BLOCK of UNROLL
// float4s (`A[4g], A[4g+1], A[4g+2], A[4g+3]` for thread g). That looks
// reasonable per-thread, but at any single (unrolled) load instruction,
// *all threads in the warp* execute that load simultaneously - and
// consecutive threads' addresses for a fixed unroll step k are
// `4g*16B` apart instead of `16B` apart, i.e. a stride-4 access pattern
// across the warp. ncu confirmed this directly: "only 16.0 of the 32
// bytes transmitted per sector are utilized... caused by a stride
// between threads" and "49.86% excessive sectors from uncoalesced
// access" - exactly the block-interleaved-vs-striped mistake.
//
// The fix: stripe the unroll across the *whole grid* instead of giving
// each thread a private contiguous block. Thread `g`'s k-th access is
// `A[g + k*totalThreads]`, not `A[g*UNROLL + k]`. At any fixed k, thread
// g's address is now exactly 16 bytes from thread g+1's (same as v2),
// fully coalesced - the only difference from v2 is that each thread now
// has UNROLL independent, non-dependent loads in flight before any
// store, which is what v3 was actually trying to test.

constexpr int UNROLL = 4;  // independent float4 loads per thread before any store

__global__ void vecAddVec4Striped(const float4* __restrict__ A, const float4* __restrict__ B,
                                   float4* __restrict__ C, size_t n4, size_t totalThreads) {
    size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    float4 a[UNROLL], b[UNROLL];
    size_t idx[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) idx[k] = g + (size_t)k * totalThreads;

#pragma unroll
    for (int k = 0; k < UNROLL; ++k) if (idx[k] < n4) a[k] = A[idx[k]];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) if (idx[k] < n4) b[k] = B[idx[k]];

    float4 c[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        c[k].x = a[k].x + b[k].x;
        c[k].y = a[k].y + b[k].y;
        c[k].z = a[k].z + b[k].z;
        c[k].w = a[k].w + b[k].w;
    }
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) if (idx[k] < n4) C[idx[k]] = c[k];
}

__global__ void vecAddTail(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, size_t start, size_t n) {
    size_t i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Note: d_input1, d_input2, d_output are device pointers
extern "C" void solution(const float* d_input1, const float* d_input2,
                          float* d_output, size_t n) {
    if (n == 0) return;
    const int threads = 256;

    size_t n4 = n / 4;              // number of float4 groups
    size_t tailStart = n4 * 4;      // scalar remainder (n % 4), same as v2

    if (n4 > 0) {
        // enough threads that each covers UNROLL strided float4 groups
        size_t threadsNeeded = (n4 + UNROLL - 1) / UNROLL;
        size_t blocks = (threadsNeeded + threads - 1) / threads;
        size_t totalThreads = blocks * threads;

        vecAddVec4Striped<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(d_input1),
            reinterpret_cast<const float4*>(d_input2),
            reinterpret_cast<float4*>(d_output), n4, totalThreads);
    }
    if (tailStart < n) {
        size_t tailN = n - tailStart;
        size_t blocks = (tailN + threads - 1) / threads;
        vecAddTail<<<blocks, threads>>>(d_input1, d_input2, d_output, tailStart, n);
    }
}
