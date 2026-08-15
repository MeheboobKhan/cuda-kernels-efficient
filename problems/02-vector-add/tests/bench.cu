// GPU benchmark + on-device correctness check for whichever solution.cu is linked.
//
//   bench.exe                  -> time the default case (n=2^30)
//   bench.exe n [iters]        -> time a specific case (iters=1 when profiling with ncu)
//   bench.exe --check          -> verify against a CPU reference, several sizes
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void solution(const float* d_input1, const float* d_input2,
                          float* d_output, size_t n);

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static void runOnDevice(const std::vector<float>& hA, const std::vector<float>& hB,
                         std::vector<float>& hC, size_t n) {
    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, n * sizeof(float)), "malloc A");
    checkCuda(cudaMalloc(&dB, n * sizeof(float)), "malloc B");
    checkCuda(cudaMalloc(&dC, n * sizeof(float)), "malloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy B");
    checkCuda(cudaMemset(dC, 0, n * sizeof(float)), "memset C");

    solution(dA, dB, dC, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "sync");

    hC.resize(n);
    checkCuda(cudaMemcpy(hC.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost), "copy C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

static int doCheck() {
    // kept small-ish so the CPU reference + host allocation stay quick;
    // deliberately includes sizes not divisible by 4 to exercise any
    // vectorized-load tail-handling path
    std::vector<size_t> cases = {1, 3, 7, 255, 1024, 1000003, 1 << 20, (1 << 20) + 5};
    int failures = 0;
    for (size_t n : cases) {
        std::vector<float> hA(n), hB(n), hC;
        for (size_t i = 0; i < n; ++i) hA[i] = (float)rand() / RAND_MAX - 0.5f;
        for (size_t i = 0; i < n; ++i) hB[i] = (float)rand() / RAND_MAX - 0.5f;

        runOnDevice(hA, hB, hC, n);

        double worst = 0.0;
        size_t worstIdx = 0;
        for (size_t i = 0; i < n; ++i) {
            double ref = (double)hA[i] + (double)hB[i];
            double e = std::fabs((double)hC[i] - ref);
            if (e > worst) { worst = e; worstIdx = i; }
        }
        bool ok = worst < 1e-5;
        std::printf("%-5s n=%9zu  max_abs_err=%.3e", ok ? "OK" : "FAIL", n, worst);
        if (!ok) {
            std::printf("  (i=%zu got=%.6f expected=%.6f)", worstIdx, hC[worstIdx],
                        (double)hA[worstIdx] + (double)hB[worstIdx]);
            ++failures;
        }
        std::printf("\n");
    }
    std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) return doCheck();

    size_t n = argc > 1 ? std::atoll(argv[1]) : (size_t)1 << 30;
    int iters = argc > 2 ? std::atoi(argv[2]) : 20;

    std::vector<float> hA(n), hB(n);
    for (auto& v : hA) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& v : hB) v = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, n * sizeof(float)), "malloc A");
    checkCuda(cudaMalloc(&dB, n * sizeof(float)), "malloc B");
    checkCuda(cudaMalloc(&dC, n * sizeof(float)), "malloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    solution(dA, dB, dC, n);  // warmup
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) solution(dA, dB, dC, n);
    cudaEventRecord(stop);
    checkCuda(cudaEventSynchronize(stop), "bench sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    // 2 reads + 1 write of 4 bytes per element, ideal-traffic bandwidth
    double bytes = 3.0 * (double)n * sizeof(float);
    double gbps = bytes / (ms * 1e-3) / 1e9;
    std::printf("n=%zu  avg=%.4f ms  (%.1f us)  %.2f GB/s\n", n, ms, ms * 1000.0, gbps);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
