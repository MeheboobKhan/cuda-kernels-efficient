#include <cuda_runtime.h>

// v3: float4 + per-thread ILP (4x unrolled float4 = 16 floats/thread).
//
// v2 (plain float4, 1 vector load/store per thread) measured identically
// to v1 on the local GTX 1650 - see benchmarks.md. That's a real result,
// but it's a GDDR5-at-128GB/s result, not a B200-at-~8TB/s result, and
// those are different regimes:
//
// - On GTX 1650, a single warp's worth of scalar 4-byte requests already
//   coalesces into full 128-byte transactions, so there's nothing left
//   for wider loads to fix - the bus is already driven at 100% with
//   scalar accesses at this bandwidth/SM-count/occupancy combination.
// - On a part with ~60x the bandwidth and (relatively) higher absolute
//   memory latency in cycles, saturating the bus requires more
//   *outstanding* memory requests in flight per SM to hide that latency
//   (Little's Law: bytes-in-flight = bandwidth x latency, and that
//   product scales up with bandwidth). Two standard levers for that:
//   wider transactions per request (float4, still worth doing - fewer,
//   bigger requests per instruction) and more independent requests
//   issued per thread before any dependent store (ILP/unrolling), so
//   the SM has enough non-dependent loads in flight to cover latency
//   even at lower occupancy.
//
// This version does both: each thread issues 4 independent float4 loads
// (16 floats) before writing any of them back, instead of one load-op
// pair at a time. This is the same pattern NVIDIA's own bandwidth
// microbenchmarks and STREAM-style CUDA ports use to reach a high
// fraction of peak on high-bandwidth parts.
//
// IMPORTANT CAVEAT: this has not been (and locally cannot be) validated
// as a real improvement - the local GTX 1650 shows no difference from
// v1/v2 (expected, see above; it's already at its own ceiling either
// way). Whether this actually beats v2 requires a real tensara
// submission on their GPU. Kept alongside v1/v2 rather than replacing
// solution.cu blindly - see benchmarks.md and README.md for the honest
// status of this version.

constexpr int UNROLL = 4;            // float4 loads per thread per pass
constexpr int ELEMS_PER_THREAD = UNROLL * 4;  // = 16 floats

__global__ void vecAddVec4Unrolled(const float4* __restrict__ A, const float4* __restrict__ B,
                                    float4* __restrict__ C, size_t n4Groups) {
    // each thread owns UNROLL consecutive float4 slots: n4Groups is the
    // number of UNROLL-sized groups; group g's float4 indices are
    // g*UNROLL .. g*UNROLL+UNROLL-1
    size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n4Groups) return;

    size_t base = g * UNROLL;
    float4 a[UNROLL], b[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) a[k] = A[base + k];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) b[k] = B[base + k];

    float4 c[UNROLL];
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
        c[k].x = a[k].x + b[k].x;
        c[k].y = a[k].y + b[k].y;
        c[k].z = a[k].z + b[k].z;
        c[k].w = a[k].w + b[k].w;
    }
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) C[base + k] = c[k];
}

// covers a run of complete float4 groups that didn't fit the unrolled pass
__global__ void vecAddVec4(const float4* __restrict__ A, const float4* __restrict__ B,
                            float4* __restrict__ C, size_t start4, size_t n4) {
    size_t i = start4 + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = A[i], b = B[i];
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        C[i] = c;
    }
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

    size_t nGroups = n / ELEMS_PER_THREAD;      // full 16-float groups
    size_t afterUnrolled = nGroups * ELEMS_PER_THREAD;

    const float4* A4 = reinterpret_cast<const float4*>(d_input1);
    const float4* B4 = reinterpret_cast<const float4*>(d_input2);
    float4* C4 = reinterpret_cast<float4*>(d_output);

    if (nGroups > 0) {
        size_t blocks = (nGroups + threads - 1) / threads;
        vecAddVec4Unrolled<<<blocks, threads>>>(A4, B4, C4, nGroups);
    }

    // remainder: fewer than 16 floats left, still >= 0 and < 16
    size_t rem = n - afterUnrolled;
    size_t rem4 = rem / 4;             // remaining full float4 groups (0..3)
    size_t start4 = afterUnrolled / 4; // == afterUnrolled/4 exactly, ELEMS_PER_THREAD%4==0
    if (rem4 > 0) {
        size_t blocks = (rem4 + threads - 1) / threads;
        vecAddVec4<<<blocks, threads>>>(A4, B4, C4, start4, start4 + rem4);
    }
    size_t tailStart = afterUnrolled + rem4 * 4;
    if (tailStart < n) {
        size_t tailN = n - tailStart;
        size_t blocks = (tailN + threads - 1) / threads;
        vecAddTail<<<blocks, threads>>>(d_input1, d_input2, d_output, tailStart, n);
    }
}
