#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

// v5: fully fused overlap-save. One kernel per block does load -> FFT ->
// spectrum multiply -> IFFT -> store, entirely in shared memory.
//
// Why: v4's ncu profile showed ~57 MB of DRAM traffic per call at N=524288,
// spread over ~11 passes (every cuFFT stage is a separate kernel that reads
// and writes the whole array). The minimum possible is ~6.5 MB - read A once
// with overlap, write C once. That 8.8x traffic gap is the entire remaining
// gap to the leaderboard; nothing else left is worth more than a few percent.
//
// Real input is packed 2-at-a-time into a half-length complex FFT
// (z[n] = x[2n] + i*x[2n+1]), so a 16384-point real transform needs only
// 8192 complex = 64KB of shared memory - exactly the Turing per-block limit.
// The pack/untangle/multiply/repack algebra including the DC+Nyquist packing
// is validated against a brute-force reference in tests/validate_fused.py.
//
// Falls back to a plain cuFFT path if the device can't give us the shared
// memory, or if K is too large for a viable block size.

namespace fused {

constexpr float kTwoPi = 6.283185307179586f;

__device__ __forceinline__ int bitrev(unsigned x, int bits) {
    return (int)(__brev(x) >> (32 - bits));
}

// x_padded[g] with the window/zero-pad folded in - no staging buffer
__device__ __forceinline__ float xAt(const float* __restrict__ A, int g,
                                      int N, int K, int r) {
    int apad = g - (K - 1);
    if (apad < 0 || apad >= N + K - 1) return 0.0f;
    int a = apad - r;
    return (a >= 0 && a < N) ? A[a] : 0.0f;
}

__device__ __forceinline__ float2 twiddle(float2 v, float c, float s) {
    return make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

// In-place iterative Cooley-Tukey DIT, bit-reversed input.
//
// Two butterfly levels are fused per shared-memory round trip: read 4
// elements, do both levels in registers, write 4 back. That is 4R+4W instead
// of 8R+8W for the same work, halving shared traffic - which the v5 profile
// identified as the binding constraint (L1/TEX at 76.6%, DRAM at 5%).
// For Mc=8192 (2^13, odd power) this is 6 fused passes + 1 trailing single
// stage = 7 round trips instead of 13.
//
// Verified bit-identical to the one-stage-at-a-time version in
// tests/validate_radix4.py for every size up to 2^13.
__device__ __forceinline__ void fftShared(float2* sh, int Mc, int logMc,
                                           int tid, int nthreads) {
    int L = 2, logL = 1;
    while (L <= Mc) {
        const int half = L >> 1;
        if (2 * L <= Mc) {
            // ---- fused double stage: levels L and 2L together ----
            const int nGroups = Mc >> 2;
            const float angA = -kTwoPi / (float)L;
            const float angB = -kTwoPi / (float)(2 * L);
            for (int g = tid; g < nGroups; g += nthreads) {
                int j = g & (half - 1);
                int start = (g >> (logL - 1)) << (logL + 1);
                int i0 = start + j, i1 = i0 + half;
                int i2 = start + L + j, i3 = i2 + half;

                float2 x0 = sh[i0], x1 = sh[i1], x2 = sh[i2], x3 = sh[i3];

                float sA, cA;
                __sincosf(angA * (float)j, &sA, &cA);
                float2 t1 = twiddle(x1, cA, sA);
                float2 t3 = twiddle(x3, cA, sA);
                float2 y0 = make_float2(x0.x + t1.x, x0.y + t1.y);
                float2 y1 = make_float2(x0.x - t1.x, x0.y - t1.y);
                float2 y2 = make_float2(x2.x + t3.x, x2.y + t3.y);
                float2 y3 = make_float2(x2.x - t3.x, x2.y - t3.y);

                float sB, cB;
                __sincosf(angB * (float)j, &sB, &cB);
                float2 u2 = twiddle(y2, cB, sB);
                // W_2L^(j+L/2) == -i * W_2L^j, so the second twiddle is just
                // a swap-and-negate - no extra sincos.
                float2 u3 = twiddle(y3, sB, -cB);

                sh[i0] = make_float2(y0.x + u2.x, y0.y + u2.y);
                sh[i2] = make_float2(y0.x - u2.x, y0.y - u2.y);
                sh[i1] = make_float2(y1.x + u3.x, y1.y + u3.y);
                sh[i3] = make_float2(y1.x - u3.x, y1.y - u3.y);
            }
            __syncthreads();
            L <<= 2;
            logL += 2;
        } else {
            // ---- trailing single stage (odd log2(Mc)) ----
            const int nButter = Mc >> 1;
            const float angStep = -kTwoPi / (float)L;
            for (int b = tid; b < nButter; b += nthreads) {
                int j = b & (half - 1);
                int i = ((b >> (logL - 1)) << logL) + j;
                int p = i + half;
                float s, c;
                __sincosf(angStep * (float)j, &s, &c);
                float2 u = sh[i];
                float2 t = twiddle(sh[p], c, s);
                sh[i] = make_float2(u.x + t.x, u.y + t.y);
                sh[p] = make_float2(u.x - t.x, u.y - t.y);
            }
            __syncthreads();
            L <<= 1;
            ++logL;
        }
    }
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// X[k] = 0.5*(Zk + conj(Zm)) - 0.5i*W(k)*(Zk - conj(Zm)),  W(k)=exp(-2pi i k/M)
__device__ __forceinline__ float2 untangleOne(float2 Zk, float2 Zm, int k, float angStepM) {
    float2 S = make_float2(Zk.x + Zm.x, Zk.y - Zm.y);
    float2 D = make_float2(Zk.x - Zm.x, Zk.y + Zm.y);
    float s, c;
    __sincosf(-angStepM * (float)k, &s, &c);
    float p = c * D.x - s * D.y;
    float q = c * D.y + s * D.x;
    return make_float2(0.5f * S.x + 0.5f * q, 0.5f * S.y - 0.5f * p);
}

// Zp[k] = 0.5*(Yk + conj(Ym)) + 0.5i*Wc(k)*(Yk - conj(Ym)), Wc(k)=exp(+2pi i k/M)
__device__ __forceinline__ float2 repackOne(float2 Yk, float2 Ym, int k, float angStepM) {
    float2 S = make_float2(Yk.x + Ym.x, Yk.y - Ym.y);
    float2 D = make_float2(Yk.x - Ym.x, Yk.y + Ym.y);
    float s, c;
    __sincosf(angStepM * (float)k, &s, &c);
    float p = c * D.x - s * D.y;
    float q = c * D.y + s * D.x;
    return make_float2(0.5f * S.x - 0.5f * q, 0.5f * S.y + 0.5f * p);
}

// 1024 threads: shared memory pins us to 1 block/SM, so thread count IS
// occupancy here. At 39 registers/thread (measured, no spilling) 1024 threads
// needs 39,936 of the 65,536-register file, so it fits and gives the full
// 32 warps/SM instead of 16.
__global__ __launch_bounds__(1024) void fusedConvBlock(
    const float* __restrict__ A, const float2* __restrict__ H,
    float* __restrict__ C, int N, int K, int r, int M, int Mc, int logMc, int hop) {

    extern __shared__ float2 sh[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int base = blockIdx.x * hop;

    // load window + R2C pack, straight into bit-reversed position
    for (int n = tid; n < Mc; n += nthreads) {
        float re = xAt(A, base + 2 * n, N, K, r);
        float im = xAt(A, base + 2 * n + 1, N, K, r);
        sh[bitrev(n, logMc)] = make_float2(re, im);
    }
    __syncthreads();

    fftShared(sh, Mc, logMc, tid, nthreads);

    // untangle -> multiply by H -> repack, pairwise (k, Mc-k).
    // Each thread owns both slots of its pair exclusively, so no barrier
    // is needed between the reads and the writes.
    const int kh = Mc >> 1;
    const float angStepM = kTwoPi / (float)M;   // hoisted out of the k loop
    for (int k = tid; k < kh; k += nthreads) {
        if (k == 0) {
            // slot 0 carries DC and Nyquist, both purely real
            float2 Z0 = sh[0];
            float X0 = Z0.x + Z0.y;
            float XN = Z0.x - Z0.y;
            float Y0 = X0 * H[0].x;
            float YN = XN * H[Mc].x;
            sh[0] = make_float2(0.5f * (Y0 + YN), 0.5f * (Y0 - YN));
        } else {
            int m = Mc - k;
            float2 Zk = sh[k], Zm = sh[m];
            float2 Xk = untangleOne(Zk, Zm, k, angStepM);
            float2 Xm = untangleOne(Zm, Zk, m, angStepM);
            float2 Yk = cmul(Xk, H[k]);
            float2 Ym = cmul(Xm, H[m]);
            sh[k] = repackOne(Yk, Ym, k, angStepM);
            sh[m] = repackOne(Ym, Yk, m, angStepM);
        }
    }
    if (tid == 0) {  // k = Mc/2 is self-paired
        float2 Zk = sh[kh];
        float2 Xk = untangleOne(Zk, Zk, kh, angStepM);
        float2 Yk = cmul(Xk, H[kh]);
        sh[kh] = repackOne(Yk, Yk, kh, angStepM);
    }
    __syncthreads();

    // inverse via conjugate trick: ifft(Z) = conj(fft(conj(Z)))/Mc
    for (int n = tid; n < Mc; n += nthreads) sh[n].y = -sh[n].y;
    __syncthreads();
    for (int n = tid; n < Mc; n += nthreads) {
        int rn = bitrev(n, logMc);
        if (n < rn) {
            float2 t = sh[n];
            sh[n] = sh[rn];
            sh[rn] = t;
        }
    }
    __syncthreads();

    fftShared(sh, Mc, logMc, tid, nthreads);

    // unpack + write the valid (non-overlapping) region straight to C.
    // local index t maps to C[base + t - 2*(K-1)]; t < K-1 is discarded.
    const float inv = 1.0f / (float)Mc;
    const int shift = 2 * (K - 1);
    for (int n = tid; n < Mc; n += nthreads) {
        float2 z = sh[n];
        int t0 = 2 * n, t1 = t0 + 1;
        if (t0 >= K - 1) {
            int i = base + t0 - shift;
            if (i >= 0 && i < N) C[i] = z.x * inv;
        }
        if (t1 >= K - 1) {
            int i = base + t1 - shift;
            if (i >= 0 && i < N) C[i] = -z.y * inv;  // conj from the inverse trick
        }
    }
}

// H = rfft(reverse(B) zero-padded to M), computed once per call
__global__ void buildHPad(const float* __restrict__ B, float* __restrict__ Hpad,
                           int K, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    Hpad[idx] = (idx < K) ? B[K - 1 - idx] : 0.0f;
}

struct Cache {
    int M = -1;
    cufftHandle planH = 0;
    float* dHpad = nullptr;
    float2* dH = nullptr;
};
static Cache g_cache;

static int pickM(int K, int maxSharedBytes) {
    // smallest power of 2 with a usable hop whose half-complex form fits in shared
    for (int M = 64; M <= (1 << 22); M <<= 1) {
        if (M < 2 * K) continue;             // hop = M-K+1 must be a decent fraction of M
        size_t shBytes = (size_t)(M / 2) * sizeof(float2);
        if (shBytes <= (size_t)maxSharedBytes) return M;
        break;
    }
    return -1;
}

}  // namespace fused

// ---------------- fallback: plain single-FFT cuFFT path (v3) ----------------
namespace fallback {

struct Cache {
    int L = -1;
    cufftHandle planR2C = 0, planC2R = 0;
    float* dAB = nullptr;
    cufftComplex *dSpecAB = nullptr, *dSpecC = nullptr;
    float* dConv = nullptr;
};
static Cache g_fb;

static int nextFastSize(int minL) {
    for (int L = minL < 1 ? 1 : minL;; ++L) {
        int x = L;
        for (int p : {2, 3, 5, 7})
            while (x % p == 0) x /= p;
        if (x == 1) return L;
    }
}

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

__global__ void complexMul(const cufftComplex* __restrict__ s, cufftComplex* __restrict__ o,
                            int specLen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= specLen) return;
    cufftComplex a = s[i], b = s[specLen + i];
    o[i] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__global__ void extractNormalize(const float* __restrict__ conv, float* __restrict__ C,
                                  int N, int K, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    C[i] = conv[i + K - 1] / (float)L;
}

static void run(const float* A, const float* B, float* C, int N, int K, int r) {
    const int threads = 256;
    int L = nextFastSize(N + 2 * K - 2);
    if (g_fb.L != L) {
        if (g_fb.L != -1) {
            cufftDestroy(g_fb.planR2C); cufftDestroy(g_fb.planC2R);
            cudaFree(g_fb.dAB); cudaFree(g_fb.dSpecAB);
            cudaFree(g_fb.dSpecC); cudaFree(g_fb.dConv);
        }
        int specLen = L / 2 + 1;
        cudaMalloc(&g_fb.dAB, (size_t)2 * L * sizeof(float));
        cudaMalloc(&g_fb.dSpecAB, (size_t)2 * specLen * sizeof(cufftComplex));
        cudaMalloc(&g_fb.dSpecC, (size_t)specLen * sizeof(cufftComplex));
        cudaMalloc(&g_fb.dConv, (size_t)L * sizeof(float));
        int n[1] = {L};
        cufftPlanMany(&g_fb.planR2C, 1, n, nullptr, 1, L, nullptr, 1, specLen, CUFFT_R2C, 2);
        cufftPlan1d(&g_fb.planC2R, L, CUFFT_C2R, 1);
        g_fb.L = L;
    }
    int specLen = L / 2 + 1;
    padBoth<<<(2 * L + threads - 1) / threads, threads>>>(A, B, g_fb.dAB, N, K, L, r);
    cufftExecR2C(g_fb.planR2C, g_fb.dAB, g_fb.dSpecAB);
    complexMul<<<(specLen + threads - 1) / threads, threads>>>(g_fb.dSpecAB, g_fb.dSpecC, specLen);
    cufftExecC2R(g_fb.planC2R, g_fb.dSpecC, g_fb.dConv);
    extractNormalize<<<(N + threads - 1) / threads, threads>>>(g_fb.dConv, C, N, K, L);
}

}  // namespace fallback

extern "C" void solution(const float* A, const float* B, float* C, size_t N_, size_t K_) {
    using namespace fused;

    int N = (int)N_;
    int K = (int)K_;
    if (N <= 0) return;
    int r = (K - 1) / 2;

    int device = 0;
    cudaGetDevice(&device);
    int maxShared = 0;
    cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (maxShared <= 0) maxShared = 48 * 1024;

    int M = pickM(K, maxShared);
    if (M < 0) {  // no viable fused block size on this device
        fallback::run(A, B, C, N, K, r);
        return;
    }

    int Mc = M / 2;
    int logMc = 0;
    while ((1 << logMc) < Mc) ++logMc;
    int hop = M - K + 1;
    int numBlocks = (N + 2 * K - 2 + hop - 1) / hop;
    size_t shBytes = (size_t)Mc * sizeof(float2);

    if (g_cache.M != M) {
        if (g_cache.M != -1) {
            cufftDestroy(g_cache.planH);
            cudaFree(g_cache.dHpad);
            cudaFree(g_cache.dH);
        }
        cudaMalloc(&g_cache.dHpad, (size_t)M * sizeof(float));
        cudaMalloc(&g_cache.dH, (size_t)(Mc + 1) * sizeof(float2));
        cufftPlan1d(&g_cache.planH, M, CUFFT_R2C, 1);
        // must be re-set whenever the required size changes, not just once
        cudaFuncSetAttribute(fusedConvBlock, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)shBytes);
        g_cache.M = M;
    }

    buildHPad<<<(M + 255) / 256, 256>>>(B, g_cache.dHpad, K, M);
    cufftExecR2C(g_cache.planH, g_cache.dHpad, (cufftComplex*)g_cache.dH);

    fusedConvBlock<<<numBlocks, 1024, shBytes>>>(A, g_cache.dH, C, N, K, r, M, Mc, logMc, hop);
}
