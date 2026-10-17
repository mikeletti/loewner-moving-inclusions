import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(1)

def inertia(M, tol=1e-12):
    w = np.linalg.eigvalsh((M+M.T)/2); s = tol*max(1, np.abs(w).max())
    return (int((w > s).sum()), int((np.abs(w) <= s).sum()), int((w < -s).sum()))

# Edge-gradient-like setting: 2D grid graph, F = inclusion edges (rank deficient)
def grid(m):
    idx = -np.ones((m+2, m+2), int); k = 0
    for i in range(1, m+1):
        for j in range(1, m+1): idx[i, j] = k; k += 1
    rows = []; edges = []
    for i in range(0, m+2):
        for j in range(0, m+2):
            for di, dj in ((1, 0), (0, 1)):
                a, b = (i, j), (i+di, j+dj)
                if b[0] > m+1 or b[1] > m+1: continue
                ia, ib = idx[a], idx[b]
                if ia < 0 and ib < 0: continue
                r = np.zeros(k)
                if ia >= 0: r[ia] = -1
                if ib >= 0: r[ib] = 1
                rows.append(r); edges.append((a, b))
    return np.array(rows), edges, idx

m = 12
Dg, edges, idx = grid(m)
A0 = Dg.T@Dg
# inclusion: 4x4 node block in centre -> edges with both ends inside
blk = {(i, j) for i in range(5, 9) for j in range(5, 9)}
rows = [e for e, (a, b) in enumerate(edges) if a in blk and b in blk]
F = Dg[rows]; p = len(rows)
f = rng.standard_normal(A0.shape[0])
u0 = np.linalg.solve(A0, f); g = F@u0
G = F@np.linalg.solve(A0, F.T); G = (G+G.T)/2
U, s, _ = np.linalg.svd(F, full_matrices=False); q = (s > 1e-10).sum(); Q = U[:, :q]; P = Q@Q.T
gam = np.linalg.eigvalsh(Q.T@G@Q).min()
print("p", p, "q", q, "gamma", gam)
tau = 0.05*gam

def ratios(logd):
    d = 10**logd; Di = np.diag(1/d)
    D = g@np.linalg.solve(Di+G, g)
    DpI = g@np.linalg.solve(Di+G+tau*np.eye(p), g)
    DpP = g@np.linalg.solve(Di+G+tau*P, g)
    w = np.linalg.solve(Di+G, g)
    b = tau/(gam-tau)
    return (D-DpI)/D/b, (D-DpP)/D/b, tau*(w@w)/D/b

best = [0, 0, 0]
for t in range(200):
    x0 = rng.uniform(-4, 4, p)
    for k in (0, 2):
        res = minimize(lambda x: -ratios(np.clip(x, -6, 6))[k], x0, method='Nelder-Mead', options={'maxiter': 3000})
        r = ratios(np.clip(res.x, -6, 6))
        best[k] = max(best[k], r[k])
    best[1] = max(best[1], ratios(x0)[1])
print("max over Delta>0 of [exact width with tau*I, exact width with tau*P, one-solve width with tau*I] / (tau/(gamma-tau)):", best)
# scaling with contrast: alternate large/small
for c in (1, 2, 3, 4, 6, 8):
    logd = np.where(np.arange(p) % 2 == 0, c, -c).astype(float)
    print("alternating contrast 10^±%d:" % c, ["%.3g" % v for v in ratios(logd)])
