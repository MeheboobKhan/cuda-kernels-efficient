#include <cuda_runtime.h>
#include <cufft.h>

// FFT-based conv. Direct O(N*K) is a dead end for K=8191 - even a perfectly
// tiled kernel tops out in the low milliseconds. FFT turns this into
// O((N+K) log(N+K)), which is what actually gets you into microseconds.
//
// cross-correlation C[i] = sum_j A[i+j-r]*B[j] equals a *reversed-kernel*
// linear convolution, sliced: pad A with r zeros each side (Apad, len N+K-1),
// reverse B into Brev (len K, zero padded out to L), full linear conv of
// Apad and Brev has length N+2K-2, and C is the length-N window starting at
// offset K-1. Verified against a brute force reference in tests/validate_fft.py.
//
// plans + device buffers are cached across calls (keyed by FFT size L) since
// tensara times solution() over repeated calls - paying cuFFT plan setup
// cost every call would defeat the point.

namespace conv1d_fft {

struct Cache {
    int L = -1;
    cufftHandle planR2C = 0;
    cufftHandle planC2R = 0;
    float* dApad = nullptr;
    float* dBrev = nullptr;
    cufftComplex* dSpecA = nullptr;
    cufftComplex* dSpecB = nullptr;
    float* dConv = nullptr;
};

static Cache g_cache;

__global__ void zeroPadA(const float* __restrict__ A, float* __restrict__ Apad,
                          int N, int L, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;
    int a = idx - r;
    Apad[idx] = (a >= 0 && a < N) ? A[a] : 0.0f;
}

__global__ void reversePadB(const float* __restrict__ B, float* __restrict__ Brev,
                             int K, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;
    Brev[idx] = (idx < K) ? B[K - 1 - idx] : 0.0f;
}

__global__ void complexMulInplace(cufftComplex* __restrict__ specA,
                                   const cufftComplex* __restrict__ specB, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    cufftComplex a = specA[idx], b = specB[idx];
    cufftComplex r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    specA[idx] = r;
}

// cuFFT's C2R doesn't normalize, so divide by L here while we slice the window
__global__ void extractNormalize(const float* __restrict__ conv, float* __restrict__ C,
                                  int N, int K, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    C[i] = conv[i + K - 1] / (float)L;
}

// smallest L >= minL that's 2/3/5/7-smooth, cuFFT's mixed-radix kernels are
// fast for these and slow for anything with a big prime factor
static int nextFastSize(int minL) {
    for (int L = minL < 1 ? 1 : minL;; ++L) {
        int x = L;
        for (int p : {2, 3, 5, 7})
            while (x % p == 0) x /= p;
        if (x == 1) return L;
    }
}

}  // namespace conv1d_fft

extern "C" void solution(const float* A, const float* B, float* C, size_t N_, size_t K_) {
    using namespace conv1d_fft;

    int N = (int)N_;
    int K = (int)K_;
    if (N <= 0) return;
    int r = (K - 1) / 2;

    int minL = N + 2 * K - 2;
    int L = nextFastSize(minL);

    if (g_cache.L != L) {
        if (g_cache.L != -1) {
            cufftDestroy(g_cache.planR2C);
            cufftDestroy(g_cache.planC2R);
            cudaFree(g_cache.dApad);
            cudaFree(g_cache.dBrev);
            cudaFree(g_cache.dSpecA);
            cudaFree(g_cache.dSpecB);
            cudaFree(g_cache.dConv);
        }
        int specLen = L / 2 + 1;
        cudaMalloc(&g_cache.dApad, (size_t)L * sizeof(float));
        cudaMalloc(&g_cache.dBrev, (size_t)L * sizeof(float));
        cudaMalloc(&g_cache.dSpecA, (size_t)specLen * sizeof(cufftComplex));
        cudaMalloc(&g_cache.dSpecB, (size_t)specLen * sizeof(cufftComplex));
        cudaMalloc(&g_cache.dConv, (size_t)L * sizeof(float));
        cufftPlan1d(&g_cache.planR2C, L, CUFFT_R2C, 1);
        cufftPlan1d(&g_cache.planC2R, L, CUFFT_C2R, 1);
        g_cache.L = L;
    }

    const int threads = 256;
    zeroPadA<<<(L + threads - 1) / threads, threads>>>(A, g_cache.dApad, N, L, r);
    reversePadB<<<(L + threads - 1) / threads, threads>>>(B, g_cache.dBrev, K, L);

    cufftExecR2C(g_cache.planR2C, g_cache.dApad, g_cache.dSpecA);
    cufftExecR2C(g_cache.planR2C, g_cache.dBrev, g_cache.dSpecB);

    int specLen = L / 2 + 1;
    complexMulInplace<<<(specLen + threads - 1) / threads, threads>>>(
        g_cache.dSpecA, g_cache.dSpecB, specLen);

    cufftExecC2R(g_cache.planC2R, g_cache.dSpecA, g_cache.dConv);

    extractNormalize<<<(N + threads - 1) / threads, threads>>>(g_cache.dConv, C, N, K, L);
}
