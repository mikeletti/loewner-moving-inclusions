"""Targeted tests for the note. Same FD setting as lft_test.py (unit square, Dirichlet, n=95,
block s x s cells). Bounds are built on the gradient space Gsp = range(F) with projector P:
   G_pm = G_ref + E_r +- eps P,   eps = |lambda_{r+1}(E)|  (oracle tail)
Defect  D = c0 - J = g^T (Delta^{-1} + G)^{-1} g ;  D(G_+) <= D <= D(G_-) when Delta^{-1}+G_pm keep the sign of Delta^{-1}+G.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
src = open(_os.path.join(HERE, 'lft_test.py')).read().split('rng = np.random')[0]
exec(src)
p = 2*s*(s-1)
rng = np.random.default_rng(7)
G0, _, F0 = G_and_g(43, 43)

def proj_range(G):
    w, V = np.linalg.eigh(G); keep = w > 1e-10*w.max()
    return V[:, keep] @ V[:, keep].T, w[keep].min(), keep.sum()

P0, gam0, rk0 = proj_range(G0)
print(f"reference block: rank G = {rk0} (block nodes - 1 = {s*s-1}), gamma = min nonzero eig = {gam0:.4f}")

def defect_true(F, da, g):
    A = (A0 + F.T @ sps.diags(da) @ F).tocsc()
    return c0 - f @ spla.spsolve(A, f)

def Dfun(Dinv, Gx, g):
    return g @ np.linalg.solve(Dinv + Gx, g)

rows = []
positions = [(43, 43, 'centre'), (30, 50, 'interior'), (15, 20, 'mid'), (6, 40, '6 cells'), (3, 40, '3 cells'), (1, 40, '1 cell')]
print("\n(a) gradient-space constant gamma_mu and singular values of E_mu = G_mu - G_ref")
Edata = {}
for (i0, j0, lab) in positions:
    G, g, F = G_and_g(i0, j0)
    P, gam, rk = proj_range(G)
    E = G - G0
    sv = np.linalg.svd(E, compute_uv=False)
    same = np.linalg.norm(P - P0)
    yc = np.array([i0 + s/2, j0 + s/2])*h; dc = min(yc[0], yc[1], 1-yc[0], 1-yc[1]); rR = s*h/np.sqrt(2)
    Edata[(i0, j0)] = (G, g, F, E)
    print(f"  {lab:9s} ({i0:2d},{j0:2d}) rho=r_R/d_c={rR/dc:4.2f} gamma={gam:.4f} |P-P_ref|={same:.1e}  sv/||G||: " +
          " ".join(f"{v:.0e}" for v in sv[:10]))

def bounds(G, g, E, Dinv, r):
    w, V = np.linalg.eigh(E); o = np.argsort(-abs(w)); w, V = w[o], V[:, o]
    Er = (V[:, :r]*w[:r]) @ V[:, :r].T; eps = abs(w[r])
    Gm, Gp = G0 + Er - eps*P0, G0 + Er + eps*P0
    return Dfun(Dinv, Gp, g), Dfun(Dinv, Gm, g), eps

print("\n(b) stiff / soft / near-void / mixed-sign inclusions, r = 4")
cases = {
    'stiff  lognormal(0,1.5)*5': lambda: np.exp(rng.normal(0, 1.5, p))*5,
    'rigid  da=1e6'            : lambda: np.full(p, 1e6),
    'soft   da in[-0.9,-0.1]'  : lambda: rng.uniform(-0.9, -0.1, p),
    'nearvoid da=-0.99'        : lambda: np.full(p, -0.99),
    'nearvoid da=-0.999'       : lambda: np.full(p, -0.999),
    'mixed  da in[-0.5,5]'     : lambda: rng.uniform(-0.5, 5, p),
}
for (i0, j0, lab) in [(30, 50, 'interior'), (6, 40, '6 cells'), (1, 40, '1 cell')]:
    G, g, F, E = Edata[(i0, j0)]
    for name, gen in cases.items():
        da = gen(); Dinv = np.diag(1/da)
        D = defect_true(F, da, g)
        lo, hi, eps = bounds(G, g, E, Dinv, 4)
        ok = (lo <= D*(1+1e-12)+1e-14) and (D <= hi*(1+1e-12)+1e-14) if hi >= lo else False
        wth = abs(hi-lo)/abs(D)
        thm = 2*eps*(g@g)/(gam0-eps)**2/abs(D) if 'stiff' in name or 'rigid' in name else np.nan
        print(f"  {lab:8s} {name:27s} D={D: .6e}  [{lo: .8e}, {hi: .8e}] rel.width {wth:8.1e}  valid={ok}  thm-bound/|D|={thm:8.1e}")

print("\n(c) hierarchy in r at the boundary-adjacent position (1 cell), stiff lognormal")
G, g, F, E = Edata[(1, 40)]
da = np.exp(rng.normal(0, 1.5, p))*5; Dinv = np.diag(1/da); D = defect_true(F, da, g)
for r in [0, 2, 4, 8, 12, 16, 24]:
    lo, hi, eps = bounds(G, g, E, Dinv, r)
    print(f"  r={r:2d} eps={eps:.1e} rel.width={(hi-lo)/D:.1e} valid={lo <= D <= hi}")
