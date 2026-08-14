// GPU benchmark + on-device correctness check for whichever solution.cu is linked.
//
//   bench.exe                  -> time the default case (524288 8191)
//   bench.exe N K [iters]      -> time a specific case (iters=1 when profiling with ncu)
//   bench.exe --check          -> verify against a CPU reference, several sizes
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K);

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static void runOnDevice(const std::vector<float>& hA, const std::vector<float>& hB,
                         std::vector<float>& hC, size_t N, size_t K) {
    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "malloc A");
    checkCuda(cudaMalloc(&dB, K * sizeof(float)), "malloc B");
    checkCuda(cudaMalloc(&dC, N * sizeof(float)), "malloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), K * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    checkCuda(cudaMemset(dC, 0, N * sizeof(float)), "memset C");

    solution(dA, dB, dC, N, K);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "sync");

    hC.resize(N);
    checkCuda(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// brute-force reference, double accumulator
static std::vector<float> reference(const std::vector<float>& A, const std::vector<float>& B,
                                     int N, int K) {
    std::vector<float> C(N);
    int r = (K - 1) / 2;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < K; ++j) {
            int a = i + j - r;
            if (a >= 0 && a < N) s += (double)A[a] * (double)B[j];
        }
        C[i] = (float)s;
    }
    return C;
}

static int doCheck() {
    struct Case { int N, K; };
    // kept small enough that the O(N*K) CPU reference stays quick
    std::vector<Case> cases = {{1, 1}, {64, 7}, {1000, 31}, {5000, 127},
                                {9000, 8191}, {20000, 8191}, {40000, 8191}};
    int failures = 0;
    for (auto c : cases) {
        std::vector<float> hA(c.N), hB(c.K), hC;
        for (int i = 0; i < c.N; ++i) hA[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < c.K; ++i) hB[i] = (float)rand() / RAND_MAX - 0.5f;

        runOnDevice(hA, hB, hC, c.N, c.K);
        auto ref = reference(hA, hB, c.N, c.K);

        double worst = 0.0;
        int worstIdx = -1;
        for (int i = 0; i < c.N; ++i) {
            double denom = std::max(1.0, std::fabs((double)ref[i]));
            double e = std::fabs((double)hC[i] - (double)ref[i]) / denom;
            if (e > worst) { worst = e; worstIdx = i; }
        }
        // FP32 FFT accumulates more error than direct summation; 1e-3 relative
        // is the practical bar here (tensara's checkers are looser than this)
        bool ok = worst < 1e-3;
        std::printf("%-5s N=%6d K=%5d  max_rel_err=%.3e", ok ? "OK" : "FAIL", c.N, c.K, worst);
        if (!ok) {
            std::printf("  (i=%d got=%.6f ref=%.6f)", worstIdx, hC[worstIdx], ref[worstIdx]);
            ++failures;
        }
        std::printf("\n");
    }
    std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) return doCheck();

    size_t N = argc > 1 ? std::atoll(argv[1]) : 524288;
    size_t K = argc > 2 ? std::atoll(argv[2]) : 8191;
    int iters = argc > 3 ? std::atoi(argv[3]) : 20;

    std::vector<float> hA(N), hB(K);
    for (auto& v : hA) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& v : hB) v = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "malloc A");
    checkCuda(cudaMalloc(&dB, K * sizeof(float)), "malloc B");
    checkCuda(cudaMalloc(&dC, N * sizeof(float)), "malloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), K * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    solution(dA, dB, dC, N, K);  // warmup: plan creation, context init
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) solution(dA, dB, dC, N, K);
    cudaEventRecord(stop);
    checkCuda(cudaEventSynchronize(stop), "bench sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;
    std::printf("N=%zu K=%zu  avg=%.4f ms  (%.1f us)\n", N, K, ms, ms * 1000.0);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
