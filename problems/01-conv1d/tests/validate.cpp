// Host-only correctness check for the *indexing/tiling logic* used in
// solution.cu, without requiring a GPU. It re-implements the tiled kernel's
// exact addressing scheme (blockStart, halo staging, sAptr[j] mapping) in
// plain C++ and compares the result against a brute-force O(N*K) reference
// for many random (N, K) configurations, including edge cases.
//
// Build & run:
//   g++ -O2 -std=c++17 -Wall -o validate validate.cpp && ./validate
//
// This does not exercise CUDA-specific code paths (constant memory,
// cudaFuncSetAttribute, actual shared memory), but any bug in the halo /
// tile index math -- the main source of off-by-one errors in a kernel like
// this -- will reproduce here exactly, since the arithmetic is copied
// verbatim from solution.cu.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Brute-force reference matching the problem statement exactly:
// C[i] = sum_{j=0}^{K-1} A[i + j - r] * B[j], r = (K-1)/2, zero padding.
static std::vector<float> reference(const std::vector<float>& A,
                                     const std::vector<float>& B, int N,
                                     int K) {
    std::vector<float> C(N, 0.0f);
    const int r = (K - 1) / 2;
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;  // accumulate in double for a tighter tolerance
        for (int j = 0; j < K; ++j) {
            const int a = i + j - r;
            if (a >= 0 && a < N) sum += (double)A[a] * (double)B[j];
        }
        C[i] = (float)sum;
    }
    return C;
}

// Emulates conv1d_tiled<...> from solution.cu, one block at a time, using a
// plain std::vector<float> in place of shared memory. Mirrors the exact
// index arithmetic used on-device.
static std::vector<float> emulate_tiled(const std::vector<float>& A,
                                         const std::vector<float>& B, int N,
                                         int K, int TILE) {
    std::vector<float> C(N, 0.0f);
    const int r = (K - 1) / 2;
    const int haloSize = TILE + K - 1;
    std::vector<float> sA(haloSize);

    for (int blockStart = 0; blockStart < N; blockStart += TILE) {
        // Stage halo (this loop mirrors the strided cooperative load; here
        // we just do it serially since block execution order doesn't
        // matter for correctness).
        for (int idx = 0; idx < haloSize; ++idx) {
            const int gIdx = blockStart - r + idx;
            sA[idx] = (gIdx >= 0 && gIdx < N) ? A[gIdx] : 0.0f;
        }
        for (int t = 0; t < TILE; ++t) {
            const int i = blockStart + t;
            if (i >= N) break;
            double sum = 0.0;
            const float* sAptr = sA.data() + t;
            for (int j = 0; j < K; ++j) sum += (double)sAptr[j] * (double)B[j];
            C[i] = (float)sum;
        }
    }
    return C;
}

static bool close(const std::vector<float>& a, const std::vector<float>& b,
                   float tol, int& badIdx) {
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float scale = std::max(1.0f, std::fabs(b[i]));
        if (diff > tol * scale) {
            badIdx = (int)i;
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    struct Case {
        int N, K, TILE;
    };
    std::vector<Case> cases = {
        {10, 1, 4},     {10, 3, 4},      {17, 5, 8},     {32, 7, 8},
        {33, 7, 8},     {100, 9, 16},    {100, 9, 32},   {257, 15, 64},
        {1000, 31, 128}, {2000, 63, 256}, {5000, 127, 256},
        {8200, 8191, 1024},  // mirrors the real problem's K, small N
        {70000, 8191, 1024}, // scaled-down version of the real test cases
        {1, 1, 4},
    };

    int failures = 0;
    for (const auto& c : cases) {
        std::vector<float> A(c.N), B(c.K);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        auto ref = reference(A, B, c.N, c.K);
        auto got = emulate_tiled(A, B, c.N, c.K, c.TILE);

        int badIdx = -1;
        if (!close(got, ref, 1e-3f, badIdx)) {
            std::printf(
                "FAIL  N=%d K=%d TILE=%d  first mismatch at i=%d: got=%f "
                "ref=%f\n",
                c.N, c.K, c.TILE, badIdx, got[badIdx], ref[badIdx]);
            ++failures;
        } else {
            std::printf("OK    N=%d K=%d TILE=%d\n", c.N, c.K, c.TILE);
        }
    }

    if (failures == 0) {
        std::printf("\nAll %zu cases passed.\n", cases.size());
        return 0;
    } else {
        std::printf("\n%d/%zu cases FAILED.\n", failures, cases.size());
        return 1;
    }
}
