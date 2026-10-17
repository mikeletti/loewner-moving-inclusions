"""Multi-tile many-query benchmark (note v0.6): exact tile Woodbury vs Loewner bounds, end-to-end cost.

Problem: seven-point FD on (0,1)^3, n^3 unknowns, edge coefficient a0, smooth load (as bench3d_v3d.py).
Tiles: t^3 cubes of side 0.1 centred at (0.5,0.5,0.5). Each query: random centre in a tile, inclusion =
edges within 0.1 of the centre, coefficient perturbation from one of six classes:
  0 stiff lognormal, 1 soft [-0.9,-0.1]a0, 2 mixed [-0.5,5]a0, 3 rigid 1e6,
  4 near-void [-0.999,-0.98]a0, 5 harsh mixed [-0.95,50]a0.
Costs: one sparse LU of A0 (shared); per tile: shared local geometry, exact precompute (|B| global solves),
Loewner precompute for each configuration (k global solves + local dense algebra); per query: online times.
Reference: exact Woodbury value D = g^T (Delta^{-1} + G_mu)^{-1} g with the exact tile G; a few queries are also
checked against a sparse direct solve of the perturbed system. Bounds are reported on D = c0 - J.
Containment = signed margins >= -1e-12 |D| (a floating-point test criterion, not a proof).
Usage: python bench3d_multitile.py n t Q [k1 r1 [k2 r2 ...]]
   e.g. python bench3d_multitile.py 44 2 25 16 200 64 600
Requires loewner_online.py in the same folder. Single-threaded BLAS recommended."""
import sys, json, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, scipy.linalg as sla
from scipy.linalg import lapack
from time import perf_counter as pc
from loewner_online import qd_upper_bound

n, t_side, Q = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg_args = [int(x) for x in sys.argv[4:]] or [16, 200, 64, 600]
CFGS = [(cfg_args[i], cfg_args[i+1]) for i in range(0, len(cfg_args), 2)]
CLASSES = ['stiff', 'soft', 'mixed', 'rigid', 'nearvoid', 'harsh']
tag = f"n{n}_t{t_side}_Q{Q}_" + "_".join(f"k{k}r{r}" for k, r in CFGS)
h = 1.0/(n+1); N = n**3

# ---------------- assembly (vectorized) ----------------
t0 = pc()
rows, cols, vals, mids, ends = [], [], [], [], []; e0 = 0
for ax in range(3):
    I_, J_, K_ = np.meshgrid(np.arange(-1, n), np.arange(n), np.arange(n), indexing='ij')
    P = [J_, K_]; P.insert(ax, I_); P = [x.ravel() for x in P]
    Qn = [x.copy() for x in P]; Qn[ax] = Qn[ax] + 1
    ne = P[0].size; eid = e0 + np.arange(ne)
    idxP = (P[0]*n + P[1])*n + P[2]; idxQ = (Qn[0]*n + Qn[1])*n + Qn[2]
    okP = (P[ax] >= 0); okQ = (Qn[ax] < n)
    rows += [eid[okP], eid[okQ]]; cols += [idxP[okP], idxQ[okQ]]; vals += [np.full(okP.sum(), -1.0/h), np.full(okQ.sum(), 1.0/h)]
    mids.append(np.stack([(P[d] + 1 + (0.5 if d == ax else 0))*h for d in range(3)], 1))
    ends.append(np.stack([np.where(okP, idxP, -1), np.where(okQ, idxQ, -1)], 1)); e0 += ne
D = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(e0, N))
mid = np.concatenate(mids); en = np.concatenate(ends)
a0 = 1.0 + 0.4*np.sin(3*mid[:, 0])*np.cos(2*mid[:, 1]) + 0.3*mid[:, 2]
A0 = (D.T @ sps.diags(a0) @ D).tocsc()
X = np.stack(np.meshgrid(*(np.arange(1, n+1)*h,)*3, indexing='ij'), -1).reshape(-1, 3)
f = np.exp(-20*((X[:, 0]-0.35)**2 + (X[:, 1]-0.6)**2 + (X[:, 2]-0.5)**2)) + 0.3
t_asm = pc() - t0
t0 = pc(); lu0 = spla.splu(A0); t_lu = pc() - t0
u0 = lu0.solve(f); c0 = float(f @ u0); gfull = D @ u0
# node -> incident edges (CSR of |D|^T)
inc = (abs(D).T).tocsr()

_lw = {}
def sytrf(Nm):
    p = Nm.shape[0]
    if p not in _lw: _lw[p] = int(lapack.dsytrf_lwork(p, lower=1)[0])
    lu, ipiv, info = lapack.dsytrf(Nm, lower=1, lwork=_lw[p], overwrite_a=1); return lu, ipiv
def sytrs(fac, rhs): return lapack.dsytrs(fac[0], fac[1], rhs, lower=1)[0]
def neg_inertia(fac):
    lu, ipiv = fac; d1 = ipiv > 0; neg = int(np.sum(np.diag(lu)[d1] < 0)); st = np.where(ipiv < 0)[0][::2]
    if len(st):
        a_, b_, c_ = lu[st, st], lu[st+1, st], lu[st+1, st+1]; det = a_*c_ - b_*b_
        neg += int(np.sum(det < 0) + 2*np.sum((det >= 0) & (a_ + c_ < 0)))
    return neg
def dbox(c, lo, hi):
    dd = np.maximum(np.maximum(lo-c, 0), np.maximum(c-hi, 0)); return np.linalg.norm(dd, axis=1)

r_inc, ell = 0.10, 0.10
centres = [0.5 + 0.1*(i - (t_side-1)/2) for i in range(t_side)]
tiles = [(np.array([cx, cy, cz]) - 0.05, np.array([cx, cy, cz]) + 0.05) for cx in centres for cy in centres for cz in centres]
rng_q = np.random.default_rng(11)

def make_queries(lo, hi):
    qs = []
    for _ in range(Q):
        mu = lo + rng_q.uniform(0, 1, 3)*(hi-lo); Rm = np.where(np.linalg.norm(mid-mu, axis=1) <= r_inc)[0]; ar = a0[Rm]; p = len(Rm)
        das = [np.exp(rng_q.normal(0, 1.5, p))*5, rng_q.uniform(-0.9, -0.1, p)*ar, rng_q.uniform(-0.5, 5, p)*ar,
               np.full(p, 1e6), rng_q.uniform(-0.999, -0.98, p)*ar, rng_q.uniform(-0.95, 50, p)*ar]
        for cl, da in enumerate(das): qs.append((Rm, da, ar, cl))
    return qs

out = dict(n=n, N=N, tiles=len(tiles), Q_positions_per_tile=Q, classes=CLASSES, configs=CFGS, c0=c0,
           assembly_s=t_asm, A0_LU_s=t_lu, per_tile=[], direct_checks=[])
def dump():
    try:
        import resource; out["peak_RSS_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    except ImportError:
        try:
            import psutil; out["peak_RSS_GB"] = psutil.Process().memory_info().peak_wset/1e9
        except Exception: out["peak_RSS_GB"] = None
    json.dump(out, open(f'multitile_{tag}.json', 'w'), indent=1)

for ti, (lo, hi) in enumerate(tiles):
    rec = dict(tile=ti, lo=lo.tolist())
    # ---------- shared local geometry ----------
    t0 = pc()
    RT = np.where(dbox(mid, lo, hi) <= r_inc)[0]
    om = np.where(dbox(mid, lo, hi) <= r_inc+ell)[0]; inom = np.zeros(e0, bool); inom[om] = True
    Vom = np.unique(en[om].ravel()); Vom = Vom[Vom >= 0]
    allin = np.array([inom[inc.indices[inc.indptr[v]:inc.indptr[v+1]]].all() for v in Vom])
    I = Vom[allin]; B = Vom[~allin]
    AII = A0[I][:, I].tocsc(); AIB = A0[I][:, B].toarray(); luI = spla.splu(AII)
    E = -luI.solve(AIB)
    Dom = D[om]; Aom = (Dom.T @ sps.diags(a0[om]) @ Dom).tocsr()
    S = Aom[B][:, B].toarray() + AIB.T @ E; S = 0.5*(S+S.T)
    Din = D[RT][:, I].tocsr(); DinT = Din.T.tocsc(); G0 = np.empty((len(RT), len(RT)))
    for c_ in range(0, len(RT), 1024):
        cs = slice(c_, min(c_+1024, len(RT))); G0[:, cs] = Din @ luI.solve(DinT[:, cs].toarray())
    G0 = 0.5*(G0+G0.T); M = np.asarray(Din @ E); del E, AIB
    rec.update(nB=int(len(B)), nI=int(len(I)), rows=int(len(RT)), geometry_s=pc()-t0)
    # ---------- exact tile Woodbury ----------
    t0 = pc(); Gam = np.empty((len(B), len(B)))
    for c_ in range(0, len(B), 512):
        cs = np.arange(c_, min(c_+512, len(B))); rhs = np.zeros((N, len(cs))); rhs[B[cs], np.arange(len(cs))] = 1.0
        Gam[:, cs] = lu0.solve(rhs)[B, :]
    Gex = G0 + M @ Gam @ M.T; Gex = 0.5*(Gex+Gex.T); del Gam
    rec['exact_offline_s'] = pc()-t0
    # ---------- Loewner configurations ----------
    t0 = pc(); ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max()
    Sh = (Vs[:, keep]*np.sqrt(ws[keep])) @ Vs[:, keep].T; Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T
    Z = M @ Shp; t_eig = pc()-t0
    LW = []
    for (K, TT) in CFGS:
        t0 = pc(); rng = np.random.default_rng(0)
        Om = rng.normal(size=(Z.shape[1], K+10)); Y = Z @ Om
        for _ in range(2): Y = Z @ (Z.T @ Y)
        Qr, _ = np.linalg.qr(Y); _, sv, Vt = np.linalg.svd(Qr.T @ Z, full_matrices=False); Vk = Vt[:K].T
        W = Sh @ Vk; rhs = np.zeros((N, K)); rhs[B, :] = W; GW = lu0.solve(rhs)[B, :]
        MG = M @ GW; Gm = G0 + MG @ np.linalg.solve(W.T @ GW, MG.T); Gm = 0.5*(Gm+Gm.T)
        Xr = Z - (Z @ Vk) @ Vk.T
        Om2 = rng.normal(size=(Xr.shape[0], TT+10)); Y2 = Xr.T @ Om2
        for _ in range(2): Y2 = Xr.T @ (Xr @ Y2)
        Wt, _ = np.linalg.qr(Y2); Xt = Xr @ Wt
        Rres = Xr - Xt @ Wt.T; rho = float(np.sum(Rres*Rres)); del Rres
        tau = rho*(1 + 1e-8) + 1e-15*float(np.sum(Xr*Xr)); xt2 = float(np.linalg.norm(Xt, 2)**2); del Xr
        LW.append(dict(K=K, TT=TT, Gm=Gm, Xt=Xt, tau=tau, eta=tau+xt2, offline_s=t_eig + pc()-t0, gap=float(sv[K]**2) if len(sv) > K else 0.0))
    del Z, Sh, Shp, M, G0
    rec['loewner'] = [dict(K=c['K'], r=c['TT']+10, offline_s=c['offline_s'], tau=c['tau'], eta=c['eta'], gap=c['gap'],
                           storage_MB=(c['Gm'].nbytes + c['Xt'].nbytes)/1e6) for c in LW]
    rec['exact_storage_MB'] = Gex.nbytes/1e6
    # ---------- queries ----------
    rowmap = np.full(e0, -1); rowmap[RT] = np.arange(len(RT))
    queries = make_queries(lo, hi)
    def online_exact(ix, g, da):
        fac = sytrf(Gex[np.ix_(ix, ix)] + np.diag(1/da)); return float(g @ sytrs(fac, g))
    def online_loewner(c, ix, g, da, ar):
        Gm, Xt, tau, eta = c['Gm'], c['Xt'], c['tau'], c['eta']
        Nm = Gm[np.ix_(ix, ix)] + np.diag(1/da); Xi = Xt[ix]
        if np.all(da > 0):
            ch, info = lapack.dpotrf(Nm, lower=1); w = lapack.dpotrs(ch, g, lower=1)[0]
            Dm = float(g @ w); v = Xi.T @ w; return (Dm - float(v @ v) - tau*float(w @ w), Dm), dict(path='chol')
        P_, Q_ = da > 0, da < 0
        alpha = float(np.min(1/da[P_])) if P_.any() else np.inf
        beta = float(np.min(-1/da[Q_] - 1/ar[Q_])) - eta if Q_.any() else np.inf
        if min(alpha, beta) > 0:
            sig = min(alpha, beta); fac = sytrf(Nm.copy()); x0 = sytrs(fac, g); Dup = float(g @ x0)
            L, it, res, omg, conv = qd_upper_bound(lambda r: sytrs(fac, r), lambda v: Nm @ v + Xi @ (Xi.T @ v) + tau*v, g, x0, sig, eta, maxit=60)
            if conv: return (L, Dup), dict(path='qd' if omg == 1.0 else 'qd_damped', it=it, eta_over_sigma=eta/sig)
            fp_ = sytrf(Nm + Xi @ Xi.T + tau*np.eye(len(ix)))
            return (float(g @ sytrs(fp_, g)), Dup), dict(path='qd_fallback', it=it, eta_over_sigma=eta/sig)
        fm = sytrf(Nm.copy()); fp_ = sytrf(Nm + Xi @ Xi.T + tau*np.eye(len(ix)))
        if neg_inertia(fm) != neg_inertia(fp_): return (np.nan, np.nan), dict(path='inertia_fail')
        return (float(g @ sytrs(fp_, g)), float(g @ sytrs(fm, g))), dict(path='2fact')
    if ti == 0:  # warm-up (lwork queries, BLAS)
        Rm, da, ar, cl = queries[2]; ix = rowmap[Rm]; g = gfull[Rm]; online_exact(ix, g, da); online_loewner(LW[0], ix, g, da, ar)
    qrec = []
    for (Rm, da, ar, cl) in queries:
        ix = rowmap[Rm]; assert (ix >= 0).all(); g = gfull[Rm]
        t0 = pc(); Dex = online_exact(ix, g, da); te = pc()-t0
        item = dict(cl=cl, p=int(len(ix)), D=Dex, t_exact=te, lw=[])
        for c in LW:
            t0 = pc(); (lo_, up_), info = online_loewner(c, ix, g, da, ar); tl = pc()-t0
            fin = bool(np.isfinite(lo_) and np.isfinite(up_))
            item['lw'].append(dict(t=tl, lo=lo_ if fin else None, up=up_ if fin else None, info=info,
                                   mlo=(Dex-lo_)/abs(Dex) if fin else None, mup=(up_-Dex)/abs(Dex) if fin else None,
                                   wid=(up_-lo_)/abs(Dex) if fin else None))
        qrec.append(item)
    rec['queries'] = qrec
    # ---------- sparse direct check (tile 0, two queries) ----------
    if ti == 0:
        for (Rm, da, ar, cl) in [queries[2], queries[4]]:
            t0 = pc(); A = (A0 + D[Rm].T @ sps.diags(da) @ D[Rm]).tocsc(); luA = spla.splu(A); u = luA.solve(f); td = pc()-t0
            Dd = c0 - float(f @ u); Dw = online_exact(rowmap[Rm], gfull[Rm], da)
            out['direct_checks'].append(dict(cl=cl, direct_s=td, D_direct=Dd, D_woodbury=Dw, rel_diff=abs(Dd-Dw)/abs(Dw)))
            del luA, A
    del Gex, LW
    out['per_tile'].append(rec); dump()
    print(f"tile {ti+1}/{len(tiles)} done: geom {rec['geometry_s']:.1f}s exact {rec['exact_offline_s']:.1f}s "
          + " ".join(f"L(k={c['K']}) {c['offline_s']:.1f}s" for c in rec['loewner']), flush=True)

# ---------------- summary ----------------
allq = [q for r in out['per_tile'] for q in r['queries']]
summ = dict(total_queries=len(allq), A0_LU_s=t_lu,
            geometry_s=sum(r['geometry_s'] for r in out['per_tile']),
            exact_offline_s=sum(r['exact_offline_s'] for r in out['per_tile']),
            exact_online_mean_ms=1e3*np.mean([q['t_exact'] for q in allq]))
for j, (K, TT) in enumerate(CFGS):
    s = dict(offline_s=sum(r['loewner'][j]['offline_s'] for r in out['per_tile']),
             online_mean_ms=1e3*np.mean([q['lw'][j]['t'] for q in allq]))
    per = {}
    for cl, name in enumerate(CLASSES):
        qs = [q['lw'][j] for q in allq if q['cl'] == cl]; fin = [x for x in qs if x['wid'] is not None]
        paths = {}
        for x in qs: paths[x['info']['path']] = paths.get(x['info']['path'], 0) + 1
        per[name] = dict(n=len(qs), finite=len(fin),
                         contained=sum(1 for x in fin if x['mlo'] >= -1e-12 and x['mup'] >= -1e-12),
                         width_median=float(np.median([x['wid'] for x in fin])) if fin else None,
                         width_max=float(np.max([x['wid'] for x in fin])) if fin else None,
                         min_margin_lo=float(np.min([x['mlo'] for x in fin])) if fin else None,
                         min_margin_up=float(np.min([x['mup'] for x in fin])) if fin else None,
                         paths=paths,
                         max_eta_over_sigma=float(max([x['info'].get('eta_over_sigma', 0) for x in qs])),
                         max_iterations=int(max([x['info'].get('it', 0) for x in qs])),
                         online_mean_ms=1e3*float(np.mean([x['t'] for x in qs])),
                         exact_online_mean_ms=1e3*float(np.mean([q['t_exact'] for q in allq if q['cl'] == cl])))
    s['per_class'] = per
    s['contained_total'] = sum(v['contained'] for v in per.values()); s['finite_total'] = sum(v['finite'] for v in per.values())
    summ[f'loewner_k{K}_r{TT+10}'] = s
# end-to-end cost model from the measured totals: T(q) with q queries per tile
nt = len(tiles); curve = {}
for qpt in [10, 100, 1000, 10000]:
    nq = qpt*nt; row = dict(exact=t_lu + summ['geometry_s'] + summ['exact_offline_s'] + nq*summ['exact_online_mean_ms']/1e3)
    for j, (K, TT) in enumerate(CFGS):
        s = summ[f'loewner_k{K}_r{TT+10}']; row[f'loewner_k{K}'] = t_lu + summ['geometry_s'] + s['offline_s'] + nq*s['online_mean_ms']/1e3
    curve[str(qpt)] = row
summ['end_to_end_s_vs_queries_per_tile'] = curve
out['summary'] = summ; dump()
print(json.dumps(summ, indent=1))
