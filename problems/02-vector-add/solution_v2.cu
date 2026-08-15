#include <cuda_runtime.h>

// v2: vectorized float4 loads/stores.
//
// v1 is already memory-bandwidth bound (measured ~148 GB/s on a GTX 1650,
// right at the card's ceiling) - there's no compute to hide behind here,
// so the only lever left is how efficiently the memory system is driven.
// Issuing one 4-byte load/store per thread means 4x the instructions and
// memory transactions of issuing one 16-byte float4 per thread for the
// same amount of data - fewer, wider transactions is the standard fix
// when a kernel is already saturating bandwidth but still has scheduling/
// instruction-issue overhead to shave off.
//
// Every test size (2^20 .. 2^30) is divisible by 4, so the vectorized
// path covers them exactly - but a scalar tail handles any n not
// divisible by 4 for correctness on arbitrary input (see tests/bench.cu
// --check, which exercises this on purpose).

__global__ void vecAddVec4(const float4* __restrict__ A, const float4* __restrict__ B,
                            float4* __restrict__ C, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 a = A[i];
        float4 b = B[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
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

    size_t n4 = n / 4;          // number of full float4 groups
    size_t tailStart = n4 * 4;  // first index not covered by the vectorized pass

    if (n4 > 0) {
        size_t blocks = (n4 + threads - 1) / threads;
        vecAddVec4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(d_input1),
            reinterpret_cast<const float4*>(d_input2),
            reinterpret_cast<float4*>(d_output), n4);
    }
    if (tailStart < n) {
        size_t tailN = n - tailStart;
        size_t blocks = (tailN + threads - 1) / threads;
        vecAddTail<<<blocks, threads>>>(d_input1, d_input2, d_output, tailStart, n);
    }
}
