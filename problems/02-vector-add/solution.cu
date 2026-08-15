#include <cuda_runtime.h>

// v1: naive elementwise add, one thread per element. No tricks - this is
// the straightforward baseline every later version is measured against.
// See benchmarks.md for why (and how much) later versions beat this.

__global__ void vecAdd(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Note: d_input1, d_input2, d_output are device pointers
extern "C" void solution(const float* d_input1, const float* d_input2,
                          float* d_output, size_t n) {
    if (n == 0) return;
    const int threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_input1, d_input2, d_output, n);
}
