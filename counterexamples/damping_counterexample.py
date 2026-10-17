"""Reviewer's 2x2 counterexample (v0.5.2): undamped Richardson diverges although the a-priori
inertia hypotheses hold; the damped rule in loewner_online.qd_upper_bound converges and the
bound is valid at every iterate."""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np
import sys; sys.path.insert(0, _os.path.join(HERE, '..', 'experiments_3d'))
from loewner_online import qd_upper_bound
Delta = np.diag([100.0, -0.05]); Lam = np.eye(2)
Gm = np.diag([0.2, 0.1]); G = np.diag([0.45, 0.1]); Gp = np.diag([0.9, 0.1]); g = np.array([1.0, 0.3])
eta = np.linalg.norm(Gp - Gm, 2); alpha = 1/Delta[0, 0]; beta = -1/Delta[1, 1] - 1/Lam[1, 1] - eta; sigma = min(alpha, beta)
Nm = np.linalg.inv(Delta) + Gm; Np = np.linalg.inv(Delta) + Gp
Dp = g @ np.linalg.solve(Np, g); D = g @ np.linalg.solve(np.linalg.inv(Delta) + G, g); Dm = g @ np.linalg.solve(Nm, g)
x0 = np.linalg.solve(Nm, g)
# undamped (old rule): diverges
x = x0.copy()
for _ in range(6): x = x + np.linalg.solve(Nm, g - Np @ x)
print('undamped residual after 6 steps:', np.linalg.norm(g - Np @ x))
L, it, res, om, conv = qd_upper_bound(lambda r: np.linalg.solve(Nm, r), lambda v: Np @ v, g, x0, sigma, eta, tol=1e-13, maxit=2000)
print(f'damped: omega={om:.4f} iterations={it} converged={conv} L={L:.12f} D+={Dp:.12f}')
assert conv and L <= Dp + 1e-12 and Dp <= D <= Dm
# validity at every iterate, including the diverging undamped ones
x = x0.copy()
for _ in range(8):
    r = g - Np @ x; Lj = g @ x + x @ r - (r @ r)/sigma; assert Lj <= Dp + 1e-9*abs(Dp); x = x + np.linalg.solve(Nm, r)
print('bound valid at all undamped iterates; D+ <= D <= D-:', Dp, D, Dm)
print('OK')
