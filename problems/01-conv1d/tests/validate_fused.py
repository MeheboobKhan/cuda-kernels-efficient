import numpy as np

def reference(A, B, N, K):
    r = (K - 1) // 2
    Apad = np.zeros(N + K - 1); Apad[r:r+N] = A
    return np.array([np.dot(Apad[i:i+K], B) for i in range(N)])

def fused_overlap_save(A, B, N, K, M):
    """Mirrors the planned CUDA kernel exactly, including in-place pairwise
    spectrum handling and the DC/Nyquist special cases."""
    Mc = M // 2
    hop = M - K + 1
    r = (K - 1) // 2
    full_len = N + 2*K - 2
    num_blocks = -(-full_len // hop)

    # H computed once (cuFFT R2C of size M in the real thing)
    Hpad = np.zeros(M); Hpad[:K] = B[::-1]
    H = np.fft.rfft(Hpad)                      # length Mc+1

    def x_at(g):                                # fused window indexing, no buffer
        apad = g - (K - 1)
        if apad < 0 or apad >= N + K - 1: return 0.0
        a = apad - r
        return A[a] if 0 <= a < N else 0.0

    C = np.zeros(N)
    valid_out = np.zeros(num_blocks * hop)

    for b in range(num_blocks):
        # --- load + R2C pack (CUDA: straight into shared, bit-reversed) ---
        base = b * hop
        x = np.array([x_at(base + t) for t in range(M)])
        Z = np.fft.fft(x[0::2] + 1j * x[1::2])   # Mc-point complex FFT

        # --- untangle + multiply + repack, done pairwise in-place ---
        Zp = np.empty(Mc, dtype=complex)
        W  = lambda k: np.exp(-2j*np.pi*k/M)
        Wc = lambda k: np.exp( 2j*np.pi*k/M)

        # k = 0 carries DC and Nyquist together (both purely real)
        X0  = Z[0].real + Z[0].imag              # X[0]
        XN  = Z[0].real - Z[0].imag              # X[Mc]
        Y0  = X0 * H[0].real
        YN  = XN * H[Mc].real
        Zp[0] = complex(0.5*(Y0 + YN), 0.5*(Y0 - YN))

        # k = Mc/2 is self-paired
        kh = Mc // 2
        Zk = Z[kh]; Zmk = np.conj(Z[Mc - kh])
        Xk = 0.5*(Zk + Zmk) - 0.5j*W(kh)*(Zk - Zmk)
        Yk = Xk * H[kh]
        Ymk = np.conj(Yk)
        Zp[kh] = 0.5*(Yk + Ymk) + 0.5j*Wc(kh)*(Yk - Ymk)

        # remaining pairs (k, Mc-k): one CUDA thread owns each pair
        for k in range(1, kh):
            m = Mc - k
            Zk, Zm = Z[k], Z[m]
            Xk = 0.5*(Zk + np.conj(Zm)) - 0.5j*W(k)*(Zk - np.conj(Zm))
            Xm = 0.5*(Zm + np.conj(Zk)) - 0.5j*W(m)*(Zm - np.conj(Zk))
            Yk, Ym = Xk*H[k], Xm*H[m]
            Zp[k] = 0.5*(Yk + np.conj(Ym)) + 0.5j*Wc(k)*(Yk - np.conj(Ym))
            Zp[m] = 0.5*(Ym + np.conj(Yk)) + 0.5j*Wc(m)*(Ym - np.conj(Yk))

        # --- inverse + unpack ---
        zp = np.fft.ifft(Zp)
        y = np.empty(M); y[0::2] = zp.real; y[1::2] = zp.imag
        valid_out[b*hop:(b+1)*hop] = y[K-1:]

    for i in range(N):
        C[i] = valid_out[i + K - 1]
    return C

rng = np.random.default_rng(7)
cases = [(50,7,64),(200,15,64),(1000,31,128),(5000,127,512),(3000,1001,4096),(40000,8191,16384),(70000,8191,16384)]
worst = 0.0
for N,K,M in cases:
    A = rng.uniform(-1,1,N); B = rng.uniform(-1,1,K)
    ref = reference(A,B,N,K); got = fused_overlap_save(A,B,N,K,M)
    e = np.max(np.abs(got-ref)/np.maximum(1.0,np.abs(ref))); worst=max(worst,e)
    print(f"{'OK' if e<1e-6 else 'FAIL'}  N={N:6d} K={K:5d} M={M:6d}  err={e:.2e}")
print(f"\nworst: {worst:.2e}")

# Also verify the untangle/repack algebra in isolation, since that pair of
# formulas is where a fused R2C kernel most often goes subtly wrong.
def _algebra_check():
    M = 64; Mc = M // 2
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, M)
    z = x[0::2] + 1j * x[1::2]
    Z = np.fft.fft(z)
    k = np.arange(Mc + 1)
    W = np.exp(-2j * np.pi * k / M)
    X = 0.5*(Z[k % Mc] + np.conj(Z[(Mc-k) % Mc])) - 0.5j*W*(Z[k % Mc] - np.conj(Z[(Mc-k) % Mc]))
    print(f"untangle vs np.fft.rfft: {np.max(np.abs(X - np.fft.rfft(x))):.2e}")

_algebra_check()
