"""Inertia criterion: bounds certified iff In(Dinv+G_-) == In(Dinv+G_+) (both nonsingular).
Also check relative-width bound  W / D_- <= 2 eps / (gamma - 2 eps)  for stiff inclusions,
and Woodbury reduced form on the gradient space."""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
exec(open(_os.path.join(HERE, 'note_tests.py')).read().split('print("\\n(b)')[0])

def inertia(M):
    w = np.linalg.eigvalsh((M+M.T)/2); return (int((w > 0).sum()), int((w < 0).sum()), int((abs(w) < 1e-13).sum()))

def certify(G, g, E, Dinv, r, gam):
    w, V = np.linalg.eigh(E); o = np.argsort(-abs(w)); w, V = w[o], V[:, o]
    Er = (V[:, :r]*w[:r]) @ V[:, :r].T; eps = abs(w[r])
    Gm, Gp = G0 + Er - eps*P0, G0 + Er + eps*P0
    In_m, In_p = inertia(Dinv+Gm), inertia(Dinv+Gp)
    return Dfun(Dinv, Gp, g), Dfun(Dinv, Gm, g), eps, In_m == In_p, In_m, In_p

rng = np.random.default_rng(11)
print("inertia criterion vs actual validity")
for (i0, j0, lab) in [(30, 50, 'interior'), (6, 40, '6 cells'), (1, 40, '1 cell')]:
    G, g, F, E = Edata[(i0, j0)]
    _, gam, _ = proj_range(G)
    for name, da in [('soft[-.9,-.1]', rng.uniform(-0.9, -0.1, p)), ('void -0.99', np.full(p, -0.99)),
                     ('void -0.999', np.full(p, -0.999)), ('mixed[-.5,5]', rng.uniform(-0.5, 5, p)),
                     ('mixed[-.95,50]', rng.uniform(-0.95, 50, p))]:
        D = defect_true(F, da, g)
        for r in [4, 8, 16]:
            lo, hi, eps, same, Im, Ip = certify(G, g, E, np.diag(1/da), r, gam)
            valid = lo <= D <= hi
            print(f"  {lab:8s} {name:15s} r={r:2d} inertia equal={str(same):5s} (neg: {Im[1]:3d} vs {Ip[1]:3d})  bounds valid={str(valid):5s}  rel.width={abs(hi-lo)/abs(D):.1e}")

print("\nrelative width vs bound 2eps/(gamma-2eps), stiff lognormal, varying contrast")
for (i0, j0, lab) in [(30, 50, 'interior'), (6, 40, '6 cells'), (1, 40, '1 cell')]:
    G, g, F, E = Edata[(i0, j0)]
    _, gam, _ = proj_range(G)
    for sc in [0.01, 1, 100, 1e4]:
        da = np.exp(rng.normal(0, 1.5, p))*sc
        lo, hi, eps, same, _, _ = certify(G, g, E, np.diag(1/da), 4, gam)
        print(f"  {lab:8s} scale {sc:8.0e}: W/D_- = {(hi-lo)/hi:.2e}   bound 2eps/(gamma-2eps) = {2*eps/(gam-2*eps):.2e}   gamma={gam:.3f}")

# Woodbury reduced-form check
G, g, F, E = Edata[(30, 50)]
w, V = np.linalg.eigh(G); Q = V[:, w > 1e-10]; K = Q.T @ G @ Q; k = Q.T @ g
da = np.exp(rng.normal(0, 1.5, p))*5; B = Q.T @ np.diag(da) @ Q
print("\nreduced form check |g^T(Dinv+G)^-1 g - k^T(B^-1+K)^-1 k| / |.| =",
      abs(Dfun(np.diag(1/da), G, g) - k @ np.linalg.solve(np.linalg.inv(B)+K, k))/abs(Dfun(np.diag(1/da), G, g)),
      "  ||g - QQ^T g||/||g|| =", np.linalg.norm(g - Q @ k)/np.linalg.norm(g))
