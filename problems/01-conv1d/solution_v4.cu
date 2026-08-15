#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <algorithm>

// v4: hybrid single-FFT (v3) / overlap-save dispatch.
//
// Overlap-save only wins at N=524288 in this problem's test set - at the
// three smaller N, a single overlap-save block already covers the whole
// output (numBlocks=1), so it pays for a full block-sized transform on
// data that didn't need one, which is strictly worse than v3's
// right-sized single FFT. See benchmarks.md for the cost derivation.
// B's spectrum can't be cached across calls even though all test cases
// share K - nothing guarantees B's *content* matches just because K does,
// so it's recomputed fresh every call on both paths. Correctness first.
//
// Both paths validated against a brute-force reference in
// tests/validate_fft.py before being ported here.

namespace conv1d_fft {

static int nextFastSize(int minL) {
    for (int L = minL < 1 ? 1 : minL;; ++L) {
        int x = L;
        for (int p : {2, 3, 5, 7})
            while (x % p == 0) x /= p;
        if (x == 1) return L;
    }
}

// smallest-cost-per-valid-sample block size for overlap-save, searched
// numerically rather than hardcoded - lands around 12x K for K=8191, but
// re-derives itself if K changes
static int bestBlockSize(int K) {
    static const double mults[] = {1.2, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    int bestM = -1;
    double bestCost = 1e300;
    for (double mult : mults) {
        int M = nextFastSize((int)(mult * K));
        if (M < 64) continue;  // avoid degenerate tiny-M cases (e.g. K=1 -> M=1,
                                // where log2(M)=0 makes the cost model misfire)
        int hop = M - K + 1;
        if (hop <= 0) continue;
        double cost = ((double)M / hop) * std::log2((double)M);
        if (cost < bestCost) {
            bestCost = cost;
            bestM = M;
        }
    }
    if (bestM < 0) bestM = nextFastSize(std::max(2 * K, 64));  // safe fallback
    return bestM;
}

static double estimateSingleFFTCost(int N, int K) {
    int L = nextFastSize(N + 2 * K - 2);
    return 1.5 * (double)L * std::log2((double)L);
}

static double estimateOverlapSaveCost(int N, int K, int M) {
    int hop = M - K + 1;
    int fullLen = N + 2 * K - 2;
    int numBlocks = (fullLen + hop - 1) / hop;
    return 0.5 * (1.0 + 2.0 * numBlocks) * M * std::log2((double)M);
}

// ---------------- single-FFT path (v3) ----------------

struct CacheSingle {
    int L = -1;
    cufftHandle planR2C = 0;
    cufftHandle planC2R = 0;
    float* dAB = nullptr;
    cufftComplex* dSpecAB = nullptr;
    cufftComplex* dSpecC = nullptr;
    float* dConv = nullptr;
};
static CacheSingle g_single;

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
    cufftComplex a = specAB[idx], b = specAB[specLen + idx];
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

// ---------------- overlap-save path (v4) ----------------

struct CacheOverlapH {
    int M = -1;
    cufftHandle planH = 0;
    float* dHpad = nullptr;
    cufftComplex* dHspec = nullptr;
};
static CacheOverlapH g_overlapH;

struct CacheOverlapBlocks {
    int M = -1;
    int numBlocks = -1;
    cufftHandle planBlockR2C = 0;
    cufftHandle planBlockC2R = 0;
    float* dBlocks = nullptr;
    cufftComplex* dBlockSpecs = nullptr;
    float* dBlockConv = nullptr;
};
static CacheOverlapBlocks g_overlapBlocks;

__global__ void buildHPad(const float* __restrict__ B, float* __restrict__ Hpad, int K, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    Hpad[idx] = (idx < K) ? B[K - 1 - idx] : 0.0f;
}

// blocks[b*M + pos] = x_padded[b*hop + pos], x_padded = [K-1 zeros][Apad][zeros]
// computed on the fly, no intermediate Apad/x_padded buffer needed
__global__ void windowBlocks(const float* __restrict__ A, float* __restrict__ blocks,
                              int N, int K, int r, int M, int hop, int numBlocks) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)numBlocks * M;
    if (idx >= total) return;
    int b = (int)(idx / M);
    int pos = (int)(idx % M);
    int globalXIdx = b * hop + pos;
    int apadIdx = globalXIdx - (K - 1);
    float val = 0.0f;
    if (apadIdx >= 0 && apadIdx < N + K - 1) {
        int aIdx = apadIdx - r;
        if (aIdx >= 0 && aIdx < N) val = A[aIdx];
    }
    blocks[idx] = val;
}

__global__ void complexMulBroadcast(cufftComplex* __restrict__ blockSpecs,
                                     const cufftComplex* __restrict__ Hspec,
                                     int specLen, int numBlocks) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)numBlocks * specLen;
    if (idx >= total) return;
    int f = (int)(idx % specLen);
    cufftComplex a = blockSpecs[idx], h = Hspec[f];
    cufftComplex r;
    r.x = a.x * h.x - a.y * h.y;
    r.y = a.x * h.y + a.y * h.x;
    blockSpecs[idx] = r;
}

// C[i] = full_out[i+K-1], where full_out is the concatenation of each
// block's valid region (Yb[K-1:]) - see tests/validate_fft.py for the
// derivation of this index mapping
__global__ void stitchExtract(const float* __restrict__ blockConv, float* __restrict__ C,
                               int N, int K, int M, int hop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int fullIdx = i + K - 1;
    int b = fullIdx / hop;
    int posInValid = fullIdx % hop;
    int posInBlock = (K - 1) + posInValid;
    size_t off = (size_t)b * M + posInBlock;
    C[i] = blockConv[off] / (float)M;
}

}  // namespace conv1d_fft

extern "C" void solution(const float* A, const float* B, float* C, size_t N_, size_t K_) {
    using namespace conv1d_fft;

    int N = (int)N_;
    int K = (int)K_;
    if (N <= 0) return;
    int r = (K - 1) / 2;
    const int threads = 256;

    int M = bestBlockSize(K);
    double costSingle = estimateSingleFFTCost(N, K);
    double costOverlap = estimateOverlapSaveCost(N, K, M);

    if (costOverlap < costSingle) {
        // ---- overlap-save ----
        int hop = M - K + 1;
        int fullLen = N + 2 * K - 2;
        int numBlocks = (fullLen + hop - 1) / hop;
        int specLenBlock = M / 2 + 1;

        if (g_overlapH.M != M) {
            if (g_overlapH.M != -1) {
                cufftDestroy(g_overlapH.planH);
                cudaFree(g_overlapH.dHpad);
                cudaFree(g_overlapH.dHspec);
            }
            cudaMalloc(&g_overlapH.dHpad, (size_t)M * sizeof(float));
            cudaMalloc(&g_overlapH.dHspec, (size_t)specLenBlock * sizeof(cufftComplex));
            cufftPlan1d(&g_overlapH.planH, M, CUFFT_R2C, 1);
            g_overlapH.M = M;
        }

        if (g_overlapBlocks.M != M || g_overlapBlocks.numBlocks != numBlocks) {
            if (g_overlapBlocks.M != -1) {
                cufftDestroy(g_overlapBlocks.planBlockR2C);
                cufftDestroy(g_overlapBlocks.planBlockC2R);
                cudaFree(g_overlapBlocks.dBlocks);
                cudaFree(g_overlapBlocks.dBlockSpecs);
                cudaFree(g_overlapBlocks.dBlockConv);
            }
            cudaMalloc(&g_overlapBlocks.dBlocks, (size_t)numBlocks * M * sizeof(float));
            cudaMalloc(&g_overlapBlocks.dBlockSpecs, (size_t)numBlocks * specLenBlock * sizeof(cufftComplex));
            cudaMalloc(&g_overlapBlocks.dBlockConv, (size_t)numBlocks * M * sizeof(float));
            int n[1] = {M};
            cufftPlanMany(&g_overlapBlocks.planBlockR2C, 1, n, nullptr, 1, M,
                          nullptr, 1, specLenBlock, CUFFT_R2C, numBlocks);
            cufftPlanMany(&g_overlapBlocks.planBlockC2R, 1, n, nullptr, 1, specLenBlock,
                          nullptr, 1, M, CUFFT_C2R, numBlocks);
            g_overlapBlocks.M = M;
            g_overlapBlocks.numBlocks = numBlocks;
        }

        // B's transform: recomputed every call, on purpose (see header comment)
        buildHPad<<<(M + threads - 1) / threads, threads>>>(B, g_overlapH.dHpad, K, M);
        cufftExecR2C(g_overlapH.planH, g_overlapH.dHpad, g_overlapH.dHspec);

        size_t totalBlockElems = (size_t)numBlocks * M;
        windowBlocks<<<(totalBlockElems + threads - 1) / threads, threads>>>(
            A, g_overlapBlocks.dBlocks, N, K, r, M, hop, numBlocks);

        cufftExecR2C(g_overlapBlocks.planBlockR2C, g_overlapBlocks.dBlocks, g_overlapBlocks.dBlockSpecs);

        size_t totalSpecElems = (size_t)numBlocks * specLenBlock;
        complexMulBroadcast<<<(totalSpecElems + threads - 1) / threads, threads>>>(
            g_overlapBlocks.dBlockSpecs, g_overlapH.dHspec, specLenBlock, numBlocks);

        cufftExecC2R(g_overlapBlocks.planBlockC2R, g_overlapBlocks.dBlockSpecs, g_overlapBlocks.dBlockConv);

        stitchExtract<<<(N + threads - 1) / threads, threads>>>(
            g_overlapBlocks.dBlockConv, C, N, K, M, hop);
        return;
    }

    // ---- single FFT (v3) ----
    int minL = N + 2 * K - 2;
    int L = nextFastSize(minL);

    if (g_single.L != L) {
        if (g_single.L != -1) {
            cufftDestroy(g_single.planR2C);
            cufftDestroy(g_single.planC2R);
            cudaFree(g_single.dAB);
            cudaFree(g_single.dSpecAB);
            cudaFree(g_single.dSpecC);
            cudaFree(g_single.dConv);
        }
        int specLen = L / 2 + 1;
        cudaMalloc(&g_single.dAB, (size_t)2 * L * sizeof(float));
        cudaMalloc(&g_single.dSpecAB, (size_t)2 * specLen * sizeof(cufftComplex));
        cudaMalloc(&g_single.dSpecC, (size_t)specLen * sizeof(cufftComplex));
        cudaMalloc(&g_single.dConv, (size_t)L * sizeof(float));

        int n[1] = {L};
        cufftPlanMany(&g_single.planR2C, 1, n, nullptr, 1, L, nullptr, 1, specLen, CUFFT_R2C, 2);
        cufftPlan1d(&g_single.planC2R, L, CUFFT_C2R, 1);
        g_single.L = L;
    }

    padBoth<<<(2 * L + threads - 1) / threads, threads>>>(A, B, g_single.dAB, N, K, L, r);
    cufftExecR2C(g_single.planR2C, g_single.dAB, g_single.dSpecAB);

    int specLen = L / 2 + 1;
    complexMul<<<(specLen + threads - 1) / threads, threads>>>(g_single.dSpecAB, g_single.dSpecC, specLen);

    cufftExecC2R(g_single.planC2R, g_single.dSpecC, g_single.dConv);

    extractNormalize<<<(N + threads - 1) / threads, threads>>>(g_single.dConv, C, N, K, L);
}
