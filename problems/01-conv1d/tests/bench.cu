// actual GPU benchmark, not the CPU validator - this calls the real solution()
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K);

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    size_t N = argc > 1 ? std::atoll(argv[1]) : 524288;
    size_t K = argc > 2 ? std::atoll(argv[2]) : 8191;

    std::vector<float> hA(N), hB(K);
    for (auto& v : hA) v = (float)rand() / RAND_MAX - 0.5f;
    for (auto& v : hB) v = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "malloc A");
    checkCuda(cudaMalloc(&dB, K * sizeof(float)), "malloc B");
    checkCuda(cudaMalloc(&dC, N * sizeof(float)), "malloc C");
    checkCuda(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, hB.data(), K * sizeof(float), cudaMemcpyHostToDevice), "copy B");

    // warmup, first call pays for cudaMemcpyToSymbol + context stuff
    solution(dA, dB, dC, N, K);
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) solution(dA, dB, dC, N, K);
    cudaEventRecord(stop);
    checkCuda(cudaEventSynchronize(stop), "bench sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    double gflops = 2.0 * N * K / (ms / 1000.0) / 1e9;
    std::printf("N=%zu K=%zu  avg=%.4f ms  %.1f GFLOP/s\n", N, K, ms, gflops);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
