#include <cuda_runtime.h>
#include <cufft.h>

// FFT-based conv, v2: batched forward transform.
//
// ncu showed ~491us of a ~708us call was the FFT butterfly kernels running
// *three* times per call (R2C(A), R2C(B), C2R) - R2C(A) and R2C(B) are two
// independent same-size real transforms with no data dependency, so they
// don't need to be two separate cufftExecR2C calls. cufftPlanMany batch=2
// runs them as one call, halving the forward-transform kernel launches
// (10 -> 5) for the same total FFT work. The other ~217us was five small
// memory-bound kernels each eating a full launch + DRAM round trip for
// trivial elementwise work (all at 80-96% DRAM throughput already, i.e.
// individually bandwidth-saturated) - padBoth below folds two of those
// into one launch too.
//
// Math is identical to v1 (see tests/validate_fft.py) - this only changes
// how the same operations are scheduled on the GPU.

namespace conv1d_fft {

struct Cache {
    int L = -1;
    cufftHandle planR2C = 0;  // batched, batch=2 (A and B together)
    cufftHandle planC2R = 0;
    float* dAB = nullptr;          // [0,L) = Apad, [L,2L) = Brev
    cufftComplex* dSpecAB = nullptr;  // [0,specLen) = specA, [specLen,2*specLen) = specB
    cufftComplex* dSpecC = nullptr;   // specA * specB
    float* dConv = nullptr;        // length L, ifft output (unnormalized)
};

static Cache g_cache;

// writes Apad into AB[0,L) and reversed-padded B into AB[L,2L) in one launch
__global__ void padBoth(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ AB, int N, int K, int L, int r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * L) return;
    if (idx < L) {
        int a = idx - r;
        AB[idx] = (a >= 0 && a < N) ? A[a] : 0.0f;
    } else {
        int j = idx - L;
        AB[idx] = (j < K) ? B[K - 1 - j] : 0.0f;
    }
}

__global__ void complexMul(const cufftComplex* __restrict__ specAB,
                            cufftComplex* __restrict__ out, int specLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= specLen) return;
    cufftComplex a = specAB[idx];
    cufftComplex b = specAB[specLen + idx];
    cufftComplex r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    out[idx] = r;
}

__global__ void extractNormalize(const float* __restrict__ conv, float* __restrict__ C,
                                  int N, int K, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    C[i] = conv[i + K - 1] / (float)L;
}

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
            cudaFree(g_cache.dAB);
            cudaFree(g_cache.dSpecAB);
            cudaFree(g_cache.dSpecC);
            cudaFree(g_cache.dConv);
        }
        int specLen = L / 2 + 1;
        cudaMalloc(&g_cache.dAB, (size_t)2 * L * sizeof(float));
        cudaMalloc(&g_cache.dSpecAB, (size_t)2 * specLen * sizeof(cufftComplex));
        cudaMalloc(&g_cache.dSpecC, (size_t)specLen * sizeof(cufftComplex));
        cudaMalloc(&g_cache.dConv, (size_t)L * sizeof(float));

        int n[1] = {L};
        cufftPlanMany(&g_cache.planR2C, 1, n,
                      nullptr, 1, L,        // input: contiguous, batch stride L
                      nullptr, 1, specLen,  // output: contiguous, batch stride specLen
                      CUFFT_R2C, 2);        // batch=2: A and B together
        cufftPlan1d(&g_cache.planC2R, L, CUFFT_C2R, 1);
        g_cache.L = L;
    }

    const int threads = 256;
    padBoth<<<(2 * L + threads - 1) / threads, threads>>>(A, B, g_cache.dAB, N, K, L, r);

    cufftExecR2C(g_cache.planR2C, g_cache.dAB, g_cache.dSpecAB);

    int specLen = L / 2 + 1;
    complexMul<<<(specLen + threads - 1) / threads, threads>>>(g_cache.dSpecAB, g_cache.dSpecC, specLen);

    cufftExecC2R(g_cache.planC2R, g_cache.dSpecC, g_cache.dConv);

    extractNormalize<<<(N + threads - 1) / threads, threads>>>(g_cache.dConv, C, N, K, L);
}
