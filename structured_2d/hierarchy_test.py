"""Check: crude sandwich 0 <= G <= I/a0 reproduces the classical monotonicity inequality
   int (a0/a) da |grad u0|^2 <= J0 - J <= int da |grad u0|^2   (a = a0 + da, a0 = 1),
and compare its width with the refined hierarchy G_ref + E_r +- eps I."""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, sys
sys.argv = ['x']
exec(open(_os.path.join(HERE, 'lft_test.py')).read().split('rng = np.random')[0])
rng = np.random.default_rng(1)
p = 2*s*(s-1)
G0, _, _ = G_and_g(43, 43)
for (i0, j0) in [(30, 50), (15, 20)]:
    G, g, F = G_and_g(i0, j0)
    for scale in [0.5, 5, 500]:
        da = np.exp(rng.normal(0, 1.0, p))*scale
        J = f @ spla.spsolve((A0 + F.T @ sps.diags(da) @ F).tocsc(), f)
        d0 = c0 - J
        up = np.sum(da*g**2)                    # G_- = 0
        lo = np.sum(da/(1+da)*g**2)             # G_+ = I/a0  (a0=1)
        # direct check of the classical inequality in its usual form
        E = G - G0; w, V = np.linalg.eigh(E); o = np.argsort(-abs(w)); w, V = w[o], V[:, o]
        Er = (V[:, :4]*w[:4]) @ V[:, :4].T; eps = abs(w[4])
        Dinv = np.diag(1/da)
        lo4 = g @ np.linalg.solve(Dinv + G0 + Er + eps*np.eye(p), g)
        up4 = g @ np.linalg.solve(Dinv + G0 + Er - eps*np.eye(p), g)
        print(f"pos({i0},{j0}) contrast~{scale:6.1f}: J0-J={d0:.6e} | classical [{lo:.4e},{up:.4e}] rel.width {(up-lo)/d0:8.2e} | hierarchy r=4 [{lo4:.8e},{up4:.8e}] rel.width {(up4-lo4)/d0:8.2e}")
print("max eig G (should be <= 1/a0 = 1):", np.linalg.eigvalsh(G0).max())
