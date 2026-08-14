import numpy as np

def reference(A, B, N, K):
    # vectorized (still O(N*K), but numpy-fast enough to cover K=8191 cases
    # in the test suite without a pure-Python loop timing out)
    r = (K - 1) // 2
    Apad = np.zeros(N + K - 1)
    Apad[r:r+N] = A
    return np.array([np.dot(Apad[i:i+K], B) for i in range(N)])

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

# ---------------------------------------------------------------------------
# overlap-save (v4) - exact translation of the fused on-the-fly indexing used
# in solution.cu's windowBlocks/stitchExtract kernels (no intermediate
# Apad/x_padded arrays materialized, matching the CUDA implementation line
# for line rather than the more "obvious" array-slicing formulation).
# ---------------------------------------------------------------------------

def next_smooth(minL):
    L = max(int(minL), 1)
    while True:
        x = L
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        if x == 1:
            return L
        L += 1

def best_block_size(K):
    # mirrors solution.cu's bestBlockSize() exactly, including the M>=64
    # floor that avoids the degenerate K=1 -> M=1 -> log2(M)=0 case
    mults = [1.2, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    best_M, best_cost = None, float("inf")
    for mult in mults:
        M = next_smooth(mult * K)
        if M < 64:
            continue
        hop = M - K + 1
        if hop <= 0:
            continue
        cost = (M / hop) * np.log2(M)
        if cost < best_cost:
            best_cost, best_M = cost, M
    if best_M is None:
        best_M = next_smooth(max(2 * K, 64))
    return best_M

def overlap_save_conv(A, B, N, K, M):
    r = (K - 1) // 2
    hop = M - K + 1
    Hpad = np.zeros(M)
    Hpad[:K] = B[::-1]  # matches CUDA buildHPad
    H = np.fft.rfft(Hpad)

    full_len = N + 2 * K - 2
    num_blocks = -(-full_len // hop)  # ceil

    def x_padded_at(global_x_idx):
        # matches CUDA windowBlocks' fused indexing exactly
        apad_idx = global_x_idx - (K - 1)
        if apad_idx < 0 or apad_idx >= N + K - 1:
            return 0.0
        a_idx = apad_idx - r
        return A[a_idx] if 0 <= a_idx < N else 0.0

    blocks = np.zeros((num_blocks, M))
    for b in range(num_blocks):
        for pos in range(M):
            blocks[b, pos] = x_padded_at(b * hop + pos)

    C = np.zeros(N)
    block_conv_cache = {}
    for i in range(N):
        full_idx = i + K - 1
        b = full_idx // hop
        pos_in_valid = full_idx % hop
        pos_in_block = (K - 1) + pos_in_valid
        if b not in block_conv_cache:
            Xb = np.fft.rfft(blocks[b])
            block_conv_cache[b] = np.fft.irfft(Xb * H, n=M)
        C[i] = block_conv_cache[b][pos_in_block]
    return C

# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
cases = [
    (10, 1), (10, 3), (17, 5), (32, 7), (33, 7), (100, 9),
    (257, 15), (1000, 31), (2000, 63), (5000, 127),
    (8200, 8191), (70000, 8191), (1, 1), (524288 // 100, 8191),
]

worst = 0.0
print("-- single-FFT path (v3) --")
for N, K in cases:
    A = rng.uniform(-1, 1, N)
    B = rng.uniform(-1, 1, K)
    ref = reference(A, B, N, K)
    got = fft_conv(A, B, N, K)
    err = np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))
    worst = max(worst, err)
    status = "OK" if err < 1e-6 else "FAIL"
    print(f"{status}  N={N:6d} K={K:5d}  max_rel_err={err:.2e}")

print("\n-- overlap-save path (v4) --")
overlap_cases = [
    (10, 1), (17, 5), (100, 9), (2000, 63),
    (8200, 8191), (70000, 8191), (150000, 8191),
]
for N, K in overlap_cases:
    A = rng.uniform(-1, 1, N)
    B = rng.uniform(-1, 1, K)
    M = best_block_size(K)
    ref = reference(A, B, N, K)
    got = overlap_save_conv(A, B, N, K, M)
    err = np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))
    worst = max(worst, err)
    status = "OK" if err < 1e-6 else "FAIL"
    print(f"{status}  N={N:6d} K={K:5d} M={M:6d}  max_rel_err={err:.2e}")

print(f"\nworst relative error across all cases: {worst:.2e}")
