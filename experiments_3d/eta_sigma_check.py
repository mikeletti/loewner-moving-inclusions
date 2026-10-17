"""Recompute eta/sigma for the PC queries (n=52,60) without the expensive parts: the test queries depend only on
the mesh, a0 and the RNG seed; eta = tau + ||Xt||_2^2 is read from the v3c JSON produced on the PC."""
import sys, json, numpy as np
n = int(sys.argv[1]); d = json.load(open(sys.argv[2])); eta = d['tau'] + d['xt_norm2']
h = 1.0/(n+1)
mids = []
for ax in range(3):
    I, J, K = np.meshgrid(np.arange(-1, n), np.arange(n), np.arange(n), indexing='ij')
    P = [J, K]; P.insert(ax, I); c = [(P[dd] + 1 + (0.5 if dd == ax else 0))*h for dd in range(3)]
    mids.append(np.stack([x.ravel() for x in c], 1))
mid = np.concatenate(mids); a0 = 1.0 + 0.4*np.sin(3*mid[:, 0])*np.cos(2*mid[:, 1]) + 0.3*mid[:, 2]
lo, hi = np.array([0.45]*3), np.array([0.55]*3); r_inc = 0.10; rr = np.random.default_rng(5); out = []
for _ in range(4):
    mu = lo + rr.uniform(0, 1, 3)*(hi-lo); Rm = np.where(np.linalg.norm(mid-mu, axis=1) <= r_inc)[0]; ar = a0[Rm]
    for cl, da in enumerate([np.exp(rr.normal(0, 1.5, len(Rm)))*5, rr.uniform(-0.9, -0.1, len(Rm))*ar, rr.uniform(-0.5, 5, len(Rm))*ar, np.full(len(Rm), 1e6)]):
        if np.all(da > 0): continue
        P_, Q_ = da > 0, da < 0
        alpha = np.min(1/da[P_]) if P_.any() else np.inf; beta = np.min(-1/da[Q_] - 1/ar[Q_]) - eta
        out.append(eta/min(alpha, beta))
print(n, 'p', len(Rm), 'eta', eta, 'eta/sigma max %.2e min %.2e' % (max(out), min(out)))
