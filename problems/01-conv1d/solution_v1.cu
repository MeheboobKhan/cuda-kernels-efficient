#include <cuda_runtime.h>
#include <algorithm>

// kernel weights go in constant memory, every thread reads B[j] in lockstep
// so this broadcasts to the whole warp basically for free
constexpr int MAX_CONST_K = 16000;
__constant__ float d_kernel[MAX_CONST_K];

// one block handles TILE outputs, stages A + halo into shared mem once
// so we're not re-reading global memory K times per output
template <bool UseConstKernel>
__global__ void conv1d_tiled(const float* __restrict__ A,
                              const float* __restrict__ Bg,
                              float* __restrict__ C,
                              int N, int K, int r, int TILE) {
    extern __shared__ float sA[];

    int blockStart = blockIdx.x * TILE;
    int haloSize = TILE + K - 1;

    // cooperative load, zero pad out-of-range indices
    for (int idx = threadIdx.x; idx < haloSize; idx += blockDim.x) {
        int gIdx = blockStart - r + idx;
        sA[idx] = (gIdx >= 0 && gIdx < N) ? A[gIdx] : 0.0f;
    }
    __syncthreads();

    int i = blockStart + threadIdx.x;
    if (threadIdx.x < TILE && i < N) {
        float sum = 0.0f;
        const float* sAptr = sA + threadIdx.x; // sAptr[j] == A[i + j - r]
#pragma unroll 4
        for (int j = 0; j < K; ++j) {
            float bv = UseConstKernel ? d_kernel[j] : __ldg(&Bg[j]);
            sum += sAptr[j] * bv;
        }
        C[i] = sum;
    }
}

// fallback for when K is too big to tile (not hit by this problem's tests)
template <bool UseConstKernel>
__global__ void conv1d_direct(const float* __restrict__ A,
                               const float* __restrict__ Bg,
                               float* __restrict__ C,
                               int N, int K, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
#pragma unroll 4
    for (int j = 0; j < K; ++j) {
        int aIdx = i + j - r;
        float av = (aIdx >= 0 && aIdx < N) ? __ldg(&A[aIdx]) : 0.0f;
        float bv = UseConstKernel ? d_kernel[j] : __ldg(&Bg[j]);
        sum += av * bv;
    }
    C[i] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N_, size_t K_) {
    int N = (int)N_;
    int K = (int)K_;
    if (N <= 0) return;
    int r = (K - 1) / 2;

    int device;
    cudaGetDevice(&device);

    // figure out how much shared mem we're actually allowed to opt into
    int maxOptinShared = 0;
    cudaDeviceGetAttribute(&maxOptinShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxOptinShared <= 0) maxOptinShared = 48 * 1024;

    bool useConst = (K <= MAX_CONST_K);
    if (useConst) {
        cudaMemcpyToSymbol(d_kernel, B, K * sizeof(float));
    }

    const int blockThreads = 1024;
    int maxTileFloats = (int)(maxOptinShared / sizeof(float)) - (K - 1);

    if (maxTileFloats >= 32) {
        int TILE = std::min(blockThreads, maxTileFloats);
        size_t sharedBytes = (size_t)(TILE + K - 1) * sizeof(float);
        int gridSize = (N + TILE - 1) / TILE;

        if (useConst) {
            if (sharedBytes > 48 * 1024)
                cudaFuncSetAttribute(conv1d_tiled<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
            conv1d_tiled<true><<<gridSize, blockThreads, sharedBytes>>>(A, nullptr, C, N, K, r, TILE);
        } else {
            if (sharedBytes > 48 * 1024)
                cudaFuncSetAttribute(conv1d_tiled<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
            conv1d_tiled<false><<<gridSize, blockThreads, sharedBytes>>>(A, B, C, N, K, r, TILE);
        }
    } else {
        // K way too big for shared mem tiling, just go direct + rely on L2
        int gridSize = (N + blockThreads - 1) / blockThreads;
        if (useConst) conv1d_direct<true><<<gridSize, blockThreads>>>(A, nullptr, C, N, K, r);
        else conv1d_direct<false><<<gridSize, blockThreads>>>(A, B, C, N, K, r);
    }
}
