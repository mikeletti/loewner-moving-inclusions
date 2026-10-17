"""Effective dimension of the port-residual space for a Gram-offline robust PR certificate (one tile).
r = -M_i^T x - W y  lies in span{W, range(M^T)}.  Gram certificate needs S_ext^{-1} on that space.
We report singular values of S_ext^{-1/2} [M^T] (normalized), the numerical rank at several cutoffs, |B|,
and compare with the Löwner gap factor X_r = M S_in^{+1/2}(I - V_k V_k^T) (k=64) used with truncation rank r."""
import sys, json, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
n = int(sys.argv[1]); h = 1.0/(n+1); N = n**3
rows, cols, vals, mids, ends = [], [], [], [], []; e0 = 0
for ax in range(3):
    I_, J_, K_ = np.meshgrid(np.arange(-1, n), np.arange(n), np.arange(n), indexing='ij')
    P = [J_, K_]; P.insert(ax, I_); P = [x.ravel() for x in P]; Qn = [x.copy() for x in P]; Qn[ax] += 1
    ne = P[0].size; eid = e0 + np.arange(ne); idxP = (P[0]*n + P[1])*n + P[2]; idxQ = (Qn[0]*n + Qn[1])*n + Qn[2]
    okP = (P[ax] >= 0); okQ = (Qn[ax] < n)
    rows += [eid[okP], eid[okQ]]; cols += [idxP[okP], idxQ[okQ]]; vals += [np.full(okP.sum(), -1.0/h), np.full(okQ.sum(), 1.0/h)]
    mids.append(np.stack([(P[d] + 1 + (0.5 if d == ax else 0))*h for d in range(3)], 1))
    ends.append(np.stack([np.where(okP, idxP, -1), np.where(okQ, idxQ, -1)], 1)); e0 += ne
D = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(e0, N))
mid = np.concatenate(mids); en = np.concatenate(ends)
a0 = 1.0 + 0.4*np.sin(3*mid[:, 0])*np.cos(2*mid[:, 1]) + 0.3*mid[:, 2]
A0 = (D.T @ sps.diags(a0) @ D).tocsc(); lu0 = spla.splu(A0); inc = (abs(D).T).tocsr()
def dbox(c, lo, hi):
    dd = np.maximum(np.maximum(lo-c, 0), np.maximum(c-hi, 0)); return np.linalg.norm(dd, axis=1)
lo, hi = np.array([0.4]*3), np.array([0.5]*3)
RT = np.where(dbox(mid, lo, hi) <= 0.1)[0]; om = np.where(dbox(mid, lo, hi) <= 0.2)[0]; inom = np.zeros(e0, bool); inom[om] = True
Vom = np.unique(en[om].ravel()); Vom = Vom[Vom >= 0]
allin = np.array([inom[inc.indices[inc.indptr[v]:inc.indptr[v+1]]].all() for v in Vom]); I = Vom[allin]; B = Vom[~allin]
AII = A0[I][:, I].tocsc(); AIB = A0[I][:, B].toarray(); luI = spla.splu(AII); E = -luI.solve(AIB)
Dom = D[om]; Aom = (Dom.T @ sps.diags(a0[om]) @ Dom).tocsr(); S = Aom[B][:, B].toarray() + AIB.T @ E; S = 0.5*(S+S.T)
M = np.asarray(D[RT][:, I] @ E)
Gam = np.empty((len(B), len(B)))
for c_ in range(0, len(B), 512):
    cs = np.arange(c_, min(c_+512, len(B))); rhs = np.zeros((N, len(cs))); rhs[B[cs], np.arange(len(cs))] = 1.0; Gam[:, cs] = lu0.solve(rhs)[B, :]
Gam = 0.5*(Gam+Gam.T); S0 = np.linalg.inv(Gam); Sext = 0.5*(S0 - S + (S0 - S).T)
we, Ve = np.linalg.eigh(Sext); Sext_mh = (Ve/np.sqrt(we)) @ Ve.T
Y = Sext_mh @ M.T; sy = np.linalg.svd(Y, compute_uv=False)
ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max(); Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T
Zs = M @ Shp; U_, sz, Vt = np.linalg.svd(Zs, full_matrices=False); K = 64
Xr = Zs - (Zs @ Vt[:K].T) @ Vt[:K]; sx = np.linalg.svd(Xr, compute_uv=False)
out = dict(n=n, nB=len(B), rows=len(RT), lambda_min_Sext=float(we.min()), lambda_min_S0=float(np.linalg.eigvalsh(S0).min()))
for name, s in [('Sext^-1/2 M^T', sy), ('Loewner gap X_r (k=64)', sx)]:
    out[name] = {f'rank_rel_{c:.0e}': int(np.sum(s > c*s[0])) for c in [1e-3, 1e-4, 1e-6, 1e-8]}
    out[name]['sigma_at'] = {str(j): float(s[j]/s[0]) for j in [64, 128, 256, 400, 600] if j < len(s)}
print(json.dumps(out, indent=1))
# ---- timing: Gram-offline PR (exterior LU + rank(M)+k exterior solves) vs Loewner (k global solves) vs exact (|B| global solves)
from time import perf_counter as pc
mask = np.ones(e0); mask[om] = 0.0; Aext = (D.T @ sps.diags(a0*mask) @ D).tocsc(); EB = np.setdiff1d(np.arange(N), I); posB = np.searchsorted(EB, B)
t0 = pc(); luE = spla.splu(Aext[EB][:, EB].tocsc()); t_extlu = pc()-t0
m = int(np.sum(sy > 1e-12*sy[0])) + K
t0 = pc(); rhs = np.zeros((len(EB), m)); rhs[posB[:, None], np.arange(m)[None, :]] = np.random.default_rng(0).normal(size=(len(B), m)); luE.solve(rhs); t_ext = pc()-t0
t0 = pc(); rhs = np.zeros((N, K)); rhs[B, :] = 1.0; lu0.solve(rhs); t_glob = pc()-t0
t0 = pc(); rhs = np.zeros((N, len(B))); rhs[B, np.arange(len(B))] = 1.0; lu0.solve(rhs); t_exact = pc()-t0
print(json.dumps(dict(gram_dim=m, ext_LU_s=t_extlu, ext_solves_for_gram_s=t_ext, loewner_k_global_solves_s=t_glob, exact_B_global_solves_s=t_exact), indent=1))
