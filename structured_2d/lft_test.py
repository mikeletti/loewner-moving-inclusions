"""Sanity test of the 'moving inclusion as feedback on a fixed operator' idea.
-div((1+da)grad u) = f on unit square, Dirichlet, 5-point FD, inclusion = s x s block of cells
moving in the domain, da >= 0 arbitrary (nonaffine, high contrast) inside the block.
A(mu) = A0 + F_mu^T Delta F_mu,  A0 = -Lap_h,  F_mu = edge gradients inside the block.
Compliance J = f^T A^{-1} f = f^T A0^{-1} f - g^T (Delta^{-1} + G_mu)^{-1} g,
g = F_mu A0^{-1} f,  G_mu = F_mu A0^{-1} F_mu^T.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

n = 95; h = 1.0/(n+1); N = n*n
idx = lambda i, j: i*n + j
# edge-gradient operator D (all interior-adjacent edges, Dirichlet -> boundary values 0)
rows, cols, vals, edges = [], [], [], []
e = 0
for i in range(-1, n):          # x-edges between (i,j) and (i+1,j)
    for j in range(n):
        if i >= 0: rows.append(e); cols.append(idx(i, j)); vals.append(-1/h)
        if i+1 < n: rows.append(e); cols.append(idx(i+1, j)); vals.append(1/h)
        edges.append(('x', i, j)); e += 1
for i in range(n):              # y-edges
    for j in range(-1, n):
        if j >= 0: rows.append(e); cols.append(idx(i, j)); vals.append(-1/h)
        if j+1 < n: rows.append(e); cols.append(idx(i, j+1)); vals.append(1/h)
        edges.append(('y', i, j)); e += 1
Dg = sps.csr_matrix((vals, (rows, cols)), shape=(e, N))
A0 = (Dg.T @ Dg).tocsc()
lu = spla.splu(A0)
xs = (np.arange(n)+1)*h
X, Y = np.meshgrid(xs, xs, indexing='ij')
f = (np.exp(-60*((X-0.3)**2+(Y-0.4)**2)) + 0.5).ravel()
u0 = lu.solve(f); c0 = f @ u0

s = 10  # inclusion size in cells
def block_edges(i0, j0):
    # edges whose both endpoints lie in the block [i0,i0+s) x [j0,j0+s): local ordering independent of position
    E = []
    for di in range(s-1):
        for dj in range(s):
            E.append(('x', i0+di, j0+dj))
    for di in range(s):
        for dj in range(s-1):
            E.append(('y', i0+di, j0+dj))
    pos = {ed: k for k, ed in enumerate(edges)}
    return np.array([pos[ed] for ed in E])

def G_and_g(i0, j0):
    eid = block_edges(i0, j0)
    F = Dg[eid, :]
    Z = lu.solve(F.T.toarray())
    return F @ Z, F @ u0, F

rng = np.random.default_rng(0)
p = 2*s*(s-1)
# --- 1) G_mu is NOT low rank (eigenvalues cluster) ---
G0, g0, _ = G_and_g(43, 43)
ev = np.linalg.eigvalsh(G0)
print("1) eig(G_mu): min %.3f  max %.3f ; #eig > 0.1*max: %d of %d  (no low-rank structure)" % (ev.min(), ev.max(), (ev > 0.1*ev.max()).sum(), p))

# --- 2) G_mu - G_mu0 IS low rank (smooth boundary correction) ---
print("2) singular values of G_mu - G_ref (ref = centre), relative to ||G_ref||:")
for (i0, j0) in [(30, 50), (15, 20), (60, 70), (5, 40)]:
    G, g, _ = G_and_g(i0, j0)
    sv = np.linalg.svd(G - G0, compute_uv=False)/np.linalg.norm(G0, 2)
    dist = min(i0, j0, n-(i0+s), n-(j0+s))*h
    print("   block at (%2d,%2d), dist to boundary %.2f: sv[0..8] = %s" % (i0, j0, dist, np.array2string(sv[:9], precision=1, formatter={'float': lambda v: f'{v:.0e}'})))

# --- 3) certified two-sided compliance bounds, arbitrary nonaffine high-contrast da ---
print("3) two-sided bounds via Loewner sandwich  G_ref + E_r - eps I <= G <= G_ref + E_r + eps I")
for (i0, j0) in [(30, 50), (15, 20), (60, 70)]:
    G, g, F = G_and_g(i0, j0)
    da = np.exp(rng.normal(0, 1.5, p))*5   # lognormal stiffening, contrast up to ~1e3
    Dinv = np.diag(1/da)
    A = (A0 + F.T @ sps.diags(da) @ F).tocsc()
    J = f @ spla.spsolve(A, f)
    E = G - G0; w, V = np.linalg.eigh(E); order = np.argsort(-abs(w)); w, V = w[order], V[:, order]
    for r in [0, 2, 4, 8]:
        Er = (V[:, :r]*w[:r]) @ V[:, :r].T
        eps = abs(w[r]) if r < len(w) else 0
        Gm = G0 + Er - eps*np.eye(p); Gp = G0 + Er + eps*np.eye(p)
        Jlo = c0 - g @ np.linalg.solve(Dinv + Gm, g)
        Jhi = c0 - g @ np.linalg.solve(Dinv + Gp, g)
        print("   pos(%2d,%2d) r=%d: %.8f <= J=%.8f <= %.8f   rel.width %.1e  ok=%s" % (i0, j0, r, Jlo, J, Jhi, (Jhi-Jlo)/J, Jlo <= J+1e-12 and J <= Jhi+1e-12))

# --- 4) Kolmogorov barrier for the solution family (POD of u over positions) ---
snaps = []
for i0 in range(5, n-s-5, 4):
    for j0 in range(5, n-s-5, 4):
        _, _, F = G_and_g(i0, j0)
        da = np.full(p, 50.0)
        snaps.append(spla.spsolve((A0 + F.T @ sps.diags(da) @ F).tocsc(), f))
S = np.array(snaps).T
sv = np.linalg.svd(S - S.mean(1, keepdims=True), compute_uv=False)
sv /= sv[0]
print("4) POD of %d solution snapshots (moving inclusion, fixed contrast): sv at 1,5,10,20,40 = %s" % (S.shape[1], np.array2string(sv[[0, 4, 9, 19, 39]], precision=2)))
