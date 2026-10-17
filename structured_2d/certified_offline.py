"""Discrete certified enclosure without oracle.
Splitting:  A0^{-1}(x,y) = h^2 K(x-y) - H(x,y),  K exact lattice kernel (L K = delta on the window),
H(.,y) discrete harmonic = sum_b P(x,b) h^2 K(b-y)  (P = discrete Poisson kernel, 4n solves).
Offline: H^grad = D_d H D_d^T on edges of Omega_delta, factored as X Y^T, exact SVD via QR -> rank-r + certified tail.
Online (per position): G_hat = P0 (G^K_loc - sym(U_r[e] V_r[e]^T)) P0, eps = sigma_{r+1} (+ fp floor).
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
src = open(_os.path.join(HERE, 'lft_test.py')).read().split('rng = np.random')[0]
exec(src)            # n, h, N, idx, Dg, edges, A0, lu, f, u0, c0, s, block_edges, G_and_g
p = 2*s*(s-1)
Lmat = (A0*h*h).tocsc(); luL = spla.splu(Lmat)

# ---------- 1) exact lattice kernel K on window [-(n+1), n+1]^2 ----------
T = 2*n + 8
k1 = np.arange(T)
lam = 4 - 2*np.cos(2*np.pi*k1/T)[:, None] - 2*np.cos(2*np.pi*k1/T)[None, :]
lam[0, 0] = 1.0
Kh = 1.0/lam; Kh[0, 0] = 0.0
KT = np.real(np.fft.ifft2(Kh))
def K(dx, dy):
    dx = np.asarray(dx); dy = np.asarray(dy)
    return KT[dx % T, dy % T] - (dx**2 + dy**2)/(4.0*T*T)
# exactness check of L K = delta on the window
W = n + 2
zz = np.arange(-W, W+1)
ZX, ZY = np.meshgrid(zz, zz, indexing='ij')
LK = 4*K(ZX, ZY) - K(ZX+1, ZY) - K(ZX-1, ZY) - K(ZX, ZY+1) - K(ZX, ZY-1)
LK[W, W] -= 1.0
print(f"1) max |L K - delta| on window: {abs(LK).max():.2e}")

# ---------- 2) discrete Poisson kernel P (4n solves) and splitting check ----------
ghosts = [(-1, j) for j in range(n)] + [(n, j) for j in range(n)] + [(i, -1) for i in range(n)] + [(i, n) for i in range(n)]
nbr = lambda g: (0 if g[0] == -1 else n-1 if g[0] == n else g[0], 0 if g[1] == -1 else n-1 if g[1] == n else g[1])
B = sps.csr_matrix((np.ones(len(ghosts)), ([idx(*nbr(g)) for g in ghosts], range(len(ghosts)))), shape=(N, len(ghosts)))
t0 = time.time()
Pk = luL.solve(B.toarray())                      # N x 4n
I_, J_ = np.divmod(np.arange(N), n)
gx = np.array([g[0] for g in ghosts]); gy = np.array([g[1] for g in ghosts])
Kb = h*h*K(I_[:, None]-gx[None, :], J_[:, None]-gy[None, :])   # N x 4n : h^2 K(y - b)
print(f"2) {len(ghosts)} Poisson-kernel solves (vs N = {N}) in {time.time()-t0:.1f}s")
for y in [idx(10, 20), idx(47, 47), idx(2, 90)]:
    col_true = lu.solve(np.eye(N)[:, y])
    col_split = h*h*K(I_-I_[y], J_-J_[y]) - Pk @ Kb[y, :]
    print(f"   splitting check column {y}: rel err {np.linalg.norm(col_true-col_split)/np.linalg.norm(col_true):.1e}")

np.savez(_os.path.join(HERE, 'offline_data.npz'), Pk=Pk, Kb=Kb)
