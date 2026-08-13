// ============================================================================
// 1D Convolution (cross-correlation, zero-padded, centered kernel)
// tensara.org/problems/conv-1d
//
// C[i] = sum_{j=0}^{K-1} A[i + j - r] * B[j],   r = (K-1)/2
// (out-of-bounds reads of A are treated as 0)
//
// Design goals: minimize *global memory traffic* and maximize reuse, since
// for the given test cases (K = 8191, N up to 524288) this is a heavily
// memory-bound / reuse-bound problem (N*K ~ 4.3e9 MACs, but every element
// of A is read up to K times).
//
// Strategy
// --------
// 1. The kernel B (<= ~16000 floats = 64KB) is copied once into __constant__
//    memory. Every thread in a warp reads the same B[j] on the same loop
//    iteration (SIMT lock-step), so constant memory's broadcast path serves
//    it essentially for free (as fast as a register read once cached).
//
// 2. The input signal A is staged into on-chip shared memory per block,
//    including the "halo" (the (K-1)/2 extra elements needed on each side
//    to cover every output the block is responsible for). This converts
//    ~K reads of A per output element from *global* memory (DRAM/L2) into
//    reads from shared memory (>10x lower latency, no cache contention
//    from other SMs), and de-duplicates the redundant reads that adjacent
//    threads in the same block would otherwise all issue individually.
//
// 3. Block/tile size is chosen at launch time so that
//    (TILE + K - 1) * sizeof(float) fits in the GPU's max shared memory
//    per block (opting into the >48KB carve-out on architectures that
//    support it, e.g. Volta/Turing/Ampere/Ada/Hopper).
//
// 4. If K is so large that even a minimal tile can't fit in shared memory
//    (not the case for this problem's test cases, but handled for
//    robustness/generality), we fall back to a direct kernel that still
//    uses constant memory for B and the read-only (__ldg) cache path for
//    A, relying on L2 residency (A fits entirely in L2 on modern data-
//    center GPUs for these sizes: 524288 floats = 2MB, T4's L2 is 4MB).
//
// This file is self-contained and exposes the required
//   extern "C" void solution(const float* A, const float* B, float* C,
//                             size_t N, size_t K);
// entry point.
// ============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

namespace conv1d {

// Max kernel size we're willing to stage into constant memory.
// CUDA constant memory is 64KB total on all currently-shipping GPUs;
// 16000 floats = 62.5KB leaves a little headroom for other constants.
constexpr int kMaxConstK = 16000;

__constant__ float d_kernel[kMaxConstK];

// ---------------------------------------------------------------------------
// Tiled kernel: one block computes TILE consecutive outputs, using a shared
// memory staging buffer of size (TILE + K - 1) for the input halo.
// ---------------------------------------------------------------------------
template <bool UseConstKernel>
__global__ void conv1d_tiled(const float* __restrict__ A,
                              const float* __restrict__ Bg,
                              float* __restrict__ C, int N, int K, int r,
                              int TILE) {
    extern __shared__ float sA[];

    const int blockStart = blockIdx.x * TILE;
    const int haloSize = TILE + K - 1;

    // Cooperatively stage A[blockStart - r .. blockStart - r + haloSize - 1]
    // into shared memory, zero-padding out-of-range indices.
    for (int idx = threadIdx.x; idx < haloSize; idx += blockDim.x) {
        const int gIdx = blockStart - r + idx;
        sA[idx] = (gIdx >= 0 && gIdx < N) ? A[gIdx] : 0.0f;
    }
    __syncthreads();

    const int i = blockStart + threadIdx.x;
    if (threadIdx.x < TILE && i < N) {
        float sum = 0.0f;
        const float* sAptr = sA + threadIdx.x;  // sAptr[j] == A[i + j - r]
#pragma unroll 4
        for (int j = 0; j < K; ++j) {
            const float bv = UseConstKernel ? d_kernel[j] : __ldg(&Bg[j]);
            sum += sAptr[j] * bv;
        }
        C[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Fallback direct kernel (no shared-memory tiling). Used only when K is too
// large for any reasonably-sized tile to fit in shared memory. Still uses
// constant memory / __ldg for the (very high) reuse that L2 provides.
// ---------------------------------------------------------------------------
template <bool UseConstKernel>
__global__ void conv1d_direct(const float* __restrict__ A,
                               const float* __restrict__ Bg,
                               float* __restrict__ C, int N, int K, int r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
#pragma unroll 4
    for (int j = 0; j < K; ++j) {
        const int aIdx = i + j - r;
        const float av = (aIdx >= 0 && aIdx < N) ? __ldg(&A[aIdx]) : 0.0f;
        const float bv = UseConstKernel ? d_kernel[j] : __ldg(&Bg[j]);
        sum += av * bv;
    }
    C[i] = sum;
}

}  // namespace conv1d

extern "C" void solution(const float* A, const float* B, float* C,
                          size_t N_, size_t K_) {
    using namespace conv1d;

    const int N = static_cast<int>(N_);
    const int K = static_cast<int>(K_);
    if (N <= 0) return;
    const int r = (K - 1) / 2;

    int device = 0;
    cudaGetDevice(&device);

    int maxOptinShared = 0;
    cudaDeviceGetAttribute(&maxOptinShared,
                            cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxOptinShared <= 0) maxOptinShared = 48 * 1024;  // conservative default

    const bool useConst = (K <= kMaxConstK);
    if (useConst) {
        cudaMemcpyToSymbol(d_kernel, B, static_cast<size_t>(K) * sizeof(float));
    }

    constexpr int kBlockThreads = 1024;  // threads per block (max on all archs)
    const int maxTileFloats =
        static_cast<int>(maxOptinShared / sizeof(float)) - (K - 1);

    if (maxTileFloats >= 32) {
        // Tiled path: pick TILE <= kBlockThreads that fits shared memory.
        const int TILE = std::min(kBlockThreads, maxTileFloats);
        const size_t sharedBytes =
            static_cast<size_t>(TILE + K - 1) * sizeof(float);
        const int gridSize = (N + TILE - 1) / TILE;

        if (useConst) {
            if (sharedBytes > 48 * 1024) {
                cudaFuncSetAttribute(conv1d_tiled<true>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sharedBytes));
            }
            conv1d_tiled<true><<<gridSize, kBlockThreads, sharedBytes>>>(
                A, nullptr, C, N, K, r, TILE);
        } else {
            if (sharedBytes > 48 * 1024) {
                cudaFuncSetAttribute(conv1d_tiled<false>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sharedBytes));
            }
            conv1d_tiled<false><<<gridSize, kBlockThreads, sharedBytes>>>(
                A, B, C, N, K, r, TILE);
        }
    } else {
        // Fallback: K too large to tile usefully; go direct + rely on L2/__ldg.
        const int gridSize = (N + kBlockThreads - 1) / kBlockThreads;
        if (useConst) {
            conv1d_direct<true>
                <<<gridSize, kBlockThreads>>>(A, nullptr, C, N, K, r);
        } else {
            conv1d_direct<false>
                <<<gridSize, kBlockThreads>>>(A, B, C, N, K, r);
        }
    }
}
