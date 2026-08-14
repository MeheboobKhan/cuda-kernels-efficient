import numpy as np

def reference(A, B, N, K):
    r = (K - 1) // 2
    C = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j in range(K):
            a = i + j - r
            if 0 <= a < N:
                s += A[a] * B[j]
        C[i] = s
    return C

def next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p

def fft_conv(A, B, N, K):
    r = (K - 1) // 2
    minL = N + 2 * K - 2
    L = next_pow2(max(minL, 1))

    Apad = np.zeros(L)
    Apad[r:r+N] = A  # matches CUDA zero_pad_A: Apad[idx]=A[idx-r]

    Brev = np.zeros(L)
    Brev[:K] = B[::-1]  # matches reverse_pad_B: Brev[idx]=B[K-1-idx]

    specA = np.fft.rfft(Apad)
    specB = np.fft.rfft(Brev)
    conv = np.fft.irfft(specA * specB, n=L)  # numpy normalizes automatically

    C = conv[K-1:K-1+N]
    return C

rng = np.random.default_rng(42)
cases = [
    (10, 1), (10, 3), (17, 5), (32, 7), (33, 7), (100, 9),
    (257, 15), (1000, 31), (2000, 63), (5000, 127),
    (8200, 8191), (70000, 8191), (1, 1), (524288 // 100, 8191),
]

worst = 0.0
for N, K in cases:
    A = rng.uniform(-1, 1, N)
    B = rng.uniform(-1, 1, K)
    ref = reference(A, B, N, K)
    got = fft_conv(A, B, N, K)
    err = np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))
    worst = max(worst, err)
    status = "OK" if err < 1e-6 else "FAIL"
    print(f"{status}  N={N:6d} K={K:5d}  max_rel_err={err:.2e}")

print(f"\nworst relative error across all cases: {worst:.2e}")
