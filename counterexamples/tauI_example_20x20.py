"""Remark 4.4: adversarial maximisation (over log delta in [-6,6]) of the excess of a tau*I gap."""
import numpy as np
from scipy.optimize import minimize
from common import *
rng = np.random.default_rng(31)
m = 20
Dg, Ed = grad_op(m); A0 = Dg.T @ Dg
nodes = {(8 + a) * m + (8 + b) for a in range(4) for b in range(4)}
rows = [k for k, (a, b, _) in enumerate(Ed) if a in nodes and b in nodes]
F = Dg[rows]; p = len(rows)
G = F @ np.linalg.solve(A0, F.T); G = (G + G.T) / 2
Q = range_basis(G); gam = np.linalg.eigvalsh(Q.T @ G @ Q)[0]
for load in ['ones', 'random']:
    f = np.ones(m * m) if load == 'ones' else rng.standard_normal(m * m)
    g = F @ np.linalg.solve(A0, f)
    for tau_rel in [1e-4, 1e-2]:
        tau = tau_rel * gam; th = tau / (gam - tau)
        def ratios(x):
            dd = 10.0 ** np.clip(x, -6, 6); Di = np.diag(1 / dd)
            D = g @ np.linalg.solve(Di + G, g)
            Dp = g @ np.linalg.solve(Di + G + tau * np.eye(p), g)
            w = np.linalg.solve(Di + G, g)
            return (D - Dp) / D / th, tau * (w @ w) / D / th
        best = [0, 0]
        for k in range(2):
            for start in range(15):
                x0 = rng.uniform(-6, 6, p)
                res = minimize(lambda x: -ratios(x)[k], x0, method='Powell', options={'maxiter': 4000, 'xtol': 1e-3})
                best[k] = max(best[k], ratios(res.x)[k])
        print(f"load={load}, tau={tau_rel}*gamma (G_-=G, G_+=G+tau I): max (D_- - D_+)/D / (tau/(gamma-tau)) = {best[0]:.2f}; "
              f"max w'Ew/D / (tau/(gamma-tau)) = {best[1]:.2f}")
