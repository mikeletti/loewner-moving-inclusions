"""Löwner bounds vs certified port-reduced static condensation (PR-SC) at prescribed tolerance (note v0.7).

Same 3D problem, tiles and contrast classes as bench3d_multitile.py.
Per tile: interior nodes I (patch), port nodes B, exterior E. Port reduction keeps the patch interior EXACT
(local Woodbury with the Dirichlet patch operator G0) and restricts the port trace to z0 + span(Z), Z = Gamma W,
where W = S_in^{1/2} V_k are the transfer-operator-optimal port modes (V_k: leading right singular vectors of
M S_in^{+1/2}) and Gamma = A0^{-1}[B,B] is applied by k global solves (shared with the Löwner Nyström enrichment).
Galerkin value D_r >= D (upper bound on D = c0 - J); error D_r - D = r^T S(Delta)^{-1} r, bounded by
  coercive:  r^T Gamma r / alpha,   alpha = min(1, min_i (a0_i + delta_i)/a0_i)   (one global solve per query/level)
  robust:    r^T S_ext^{-1} r        (S(Delta) >= S_ext; one exterior solve per query/level; exterior LU per tile)
  best:      min of the two.
Löwner: levels k in KS with truncation rank r (Sections 5-7 of the note), single-factorization online algorithm.
Adaptivity: for a tolerance tol, each method tries its levels in increasing k and stops at the first one whose
bounds certify  up - lo <= tol * min(|lo|,|up|)  with lo*up > 0  (no oracle); if none does, the query is rejected.
Reported: per tolerance, certified and true-accuracy fractions, rejects, online cost of the adaptive search,
offline costs, end-to-end totals. Identity checks: D_r equals the Löwner lower-enclosure value D_- for the same k
(port-reduced Galerkin = Nyström lower bound), and D_r - D = r^T S^{-1} r (checked with the exact Gamma).
Usage: python bench3d_portred.py n t Q r [k1 k2 ...]   e.g.  python bench3d_portred.py 36 2 10 600 16 32 64 128
Requires loewner_online.py. Single-threaded BLAS recommended."""
import sys, json, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
from scipy.linalg import lapack
from time import perf_counter as pc
from loewner_online import qd_upper_bound

n, t_side, Q, TT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
KS = [int(x) for x in sys.argv[5:]] or [16, 32, 64, 128]; KMAX = max(KS)
CLASSES = ['stiff', 'soft', 'mixed', 'rigid', 'nearvoid', 'harsh']
TOLS = [1e-2, 1e-3, 1e-4, 1e-5]
tag = f"n{n}_t{t_side}_Q{Q}_r{TT}_k" + "-".join(map(str, KS))
h = 1.0/(n+1); N = n**3

# ---------------- assembly ----------------
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
inc = (abs(D).T).tocsr()
x1 = np.zeros(N); x1[N//2] = 1.0; t0 = pc(); [lu0.solve(x1) for _ in range(5)]; t_gsolve = (pc()-t0)/5

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
def certified(lo, up, tol):
    return bool(np.isfinite(lo) and np.isfinite(up) and lo*up > 0 and (up - lo) <= tol*min(abs(lo), abs(up)))

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

out = dict(n=n, N=N, tiles=len(tiles), Q_positions_per_tile=Q, classes=CLASSES, levels=KS, trunc_rank=TT+10, tols=TOLS, c0=c0,
           assembly_s=t_asm, A0_LU_s=t_lu, global_solve_s=t_gsolve, per_tile=[], direct_checks=[])
def dump():
    try:
        import resource; out["peak_RSS_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    except ImportError:
        try:
            import psutil; out["peak_RSS_GB"] = psutil.Process().memory_info().peak_wset/1e9
        except Exception: out["peak_RSS_GB"] = None
    json.dump(out, open(f'portred_{tag}.json', 'w'), indent=1)

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
    # ---------- exact tile Woodbury (also reference) ----------
    t0 = pc(); Gam = np.empty((len(B), len(B)))
    for c_ in range(0, len(B), 512):
        cs = np.arange(c_, min(c_+512, len(B))); rhs = np.zeros((N, len(cs))); rhs[B[cs], np.arange(len(cs))] = 1.0
        Gam[:, cs] = lu0.solve(rhs)[B, :]
    Gex = G0 + M @ Gam @ M.T; Gex = 0.5*(Gex+Gex.T)
    rec['exact_offline_s'] = pc()-t0          # Gam kept only for identity checks (not used by the other methods)
    # ---------- shared spectral step + k_max global solves (Löwner and PR) ----------
    t0 = pc(); ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max()
    Sh = (Vs[:, keep]*np.sqrt(ws[keep])) @ Vs[:, keep].T; Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T
    Zs = M @ Shp; rng = np.random.default_rng(0)
    Om = rng.normal(size=(Zs.shape[1], KMAX+10)); Y = Zs @ Om
    for _ in range(2): Y = Zs @ (Zs.T @ Y)
    Qr, _ = np.linalg.qr(Y); _, sv, Vt = np.linalg.svd(Qr.T @ Zs, full_matrices=False); V = Vt[:KMAX].T
    W = Sh @ V; rhs = np.zeros((N, KMAX)); rhs[B, :] = W; Z = lu0.solve(rhs)[B, :]; del rhs
    MZ = M @ Z; WZ = W.T @ Z; WZ = 0.5*(WZ + WZ.T)
    t_shared = pc()-t0
    # ---------- Löwner levels ----------
    LW = []
    for K in KS:
        t0 = pc()
        Gm = G0 + MZ[:, :K] @ np.linalg.solve(WZ[:K, :K], MZ[:, :K].T); Gm = 0.5*(Gm+Gm.T)
        Xr = Zs - (Zs @ V[:, :K]) @ V[:, :K].T
        Om2 = rng.normal(size=(Xr.shape[0], TT+10)); Y2 = Xr.T @ Om2
        for _ in range(2): Y2 = Xr.T @ (Xr @ Y2)
        Wt, _ = np.linalg.qr(Y2); Xt = Xr @ Wt
        Rres = Xr - Xt @ Wt.T; rho = float(np.sum(Rres*Rres)); del Rres
        tau = rho*(1 + 1e-8) + 1e-15*float(np.sum(Xr*Xr)); xt2 = float(np.linalg.norm(Xt, 2)**2); del Xr
        LW.append(dict(K=K, Gm=Gm, Xt=Xt, tau=tau, eta=tau+xt2, offline_s=pc()-t0))
    # ---------- PR-SC: exterior Neumann factorization ----------
    t0 = pc()
    mask = np.ones(e0); mask[om] = 0.0
    Aext = (D.T @ sps.diags(a0*mask) @ D).tocsc()
    EB = np.setdiff1d(np.arange(N), I); posB = np.searchsorted(EB, B)
    luE = spla.splu(Aext[EB][:, EB].tocsc()); del Aext
    t_ext = pc()-t0
    rec['offline'] = dict(shared_spectral_and_kmax_solves_s=t_shared, loewner_levels_s=[c['offline_s'] for c in LW],
                          pr_exterior_lu_s=t_ext, tau=[c['tau'] for c in LW], eta=[c['eta'] for c in LW],
                          storage_MB=dict(exact=Gex.nbytes/1e6, loewner=sum((c['Gm'].nbytes + c['Xt'].nbytes) for c in LW)/1e6,
                                          pr=(G0.nbytes + M.nbytes + MZ.nbytes + W.nbytes)/1e6))
    del Zs, Sh, Shp
    rowmap = np.full(e0, -1); rowmap[RT] = np.arange(len(RT))
    queries = make_queries(lo, hi)

    def online_exact(ix, g, da):
        fac = sytrf(Gex[np.ix_(ix, ix)] + np.diag(1/da)); return float(g @ sytrs(fac, g))
    def online_loewner(c, ix, g, da, ar):
        Gm, Xt, tau, eta = c['Gm'], c['Xt'], c['tau'], c['eta']
        Nm = Gm[np.ix_(ix, ix)] + np.diag(1/da); Xi = Xt[ix]
        if np.all(da > 0):
            ch, info = lapack.dpotrf(Nm, lower=1); w = lapack.dpotrs(ch, g, lower=1)[0]
            Dm = float(g @ w); v = Xi.T @ w; return (Dm - float(v @ v) - tau*float(w @ w), Dm), 'chol'
        P_, Q_ = da > 0, da < 0
        alpha = float(np.min(1/da[P_])) if P_.any() else np.inf
        beta = float(np.min(-1/da[Q_] - 1/ar[Q_])) - eta if Q_.any() else np.inf
        if min(alpha, beta) > 0:
            sig = min(alpha, beta); fac = sytrf(Nm.copy()); x0 = sytrs(fac, g); Dup = float(g @ x0)
            L, it, res, omg, conv = qd_upper_bound(lambda r: sytrs(fac, r), lambda v: Nm @ v + Xi @ (Xi.T @ v) + tau*v, g, x0, sig, eta, maxit=60)
            if conv: return (L, Dup), ('qd' if omg == 1.0 else 'qd_damped')
            fp_ = sytrf(Nm + Xi @ Xi.T + tau*np.eye(len(ix))); return (float(g @ sytrs(fp_, g)), Dup), 'qd_fallback'
        fm = sytrf(Nm.copy()); fp_ = sytrf(Nm + Xi @ Xi.T + tau*np.eye(len(ix)))
        if neg_inertia(fm) != neg_inertia(fp_): return (np.nan, np.nan), 'inertia_fail'
        return (float(g @ sytrs(fp_, g)), float(g @ sytrs(fm, g))), '2fact'
    def pr_factor(ix, g, da):
        N0 = G0[np.ix_(ix, ix)] + np.diag(1/da)
        if np.all(da > 0):
            ch, info = lapack.dpotrf(N0, lower=1); sol = lambda R: lapack.dpotrs(ch, R, lower=1)[0]
        else:
            fac = sytrf(N0); sol = lambda R: sytrs(fac, R)
        Mi = M[ix]; L = MZ[ix]; Ng = sol(g); NL = sol(L)
        return dict(Mi=Mi, L=L, Ng=Ng, NL=NL, gNg=float(g @ Ng))
    def pr_level(st, K, g):
        L, NL = st['L'][:, :K], st['NL'][:, :K]
        Kk = WZ[:K, :K] + L.T @ NL; q = -(L.T @ st['Ng'])
        y = np.linalg.solve(Kk, q); Dr = st['gNg'] - float(q @ y)
        r = -(st['Mi'].T @ (st['Ng'] + NL @ y)) - W[:, :K] @ y      # residual on the port
        return Dr, r
    def est_coercive(r, alpha):
        rhs = np.zeros(N); rhs[B] = r; return float(r @ lu0.solve(rhs)[B])/alpha
    def est_robust(r):
        rhs = np.zeros(len(EB)); rhs[posB] = r; return float(r @ luE.solve(rhs)[posB])

    if ti == 0:  # warm-up
        Rm, da, ar, cl = queries[2]; ix = rowmap[Rm]; g = gfull[Rm]
        online_exact(ix, g, da); online_loewner(LW[0], ix, g, da, ar); st = pr_factor(ix, g, da); Dr, r = pr_level(st, KS[0], g); est_coercive(r, 1.0); est_robust(r)
    qrec = []; ncheck = 0
    for (Rm, da, ar, cl) in queries:
        ix = rowmap[Rm]; assert (ix >= 0).all(); g = gfull[Rm]
        t0 = pc(); Dex = online_exact(ix, g, da); te = pc()-t0
        item = dict(cl=cl, p=int(len(ix)), D=Dex, t_exact=te, lw=[], pr=[])
        for c in LW:
            t0 = pc(); (lo_, up_), path = online_loewner(c, ix, g, da, ar); tl = pc()-t0
            item['lw'].append(dict(K=c['K'], t=tl, lo=float(lo_), up=float(up_), path=path))
        alpha = float(min(1.0, np.min(1.0 + da/ar)))
        t0 = pc(); st = pr_factor(ix, g, da); item['pr_fact_t'] = pc()-t0
        for K in KS:
            t0 = pc(); Dr, r = pr_level(st, K, g); tlev = pc()-t0
            t0 = pc(); ec = est_coercive(r, alpha); tc = pc()-t0
            t0 = pc(); er = est_robust(r); tr = pc()-t0
            lev = dict(K=K, Dr=Dr, est_coercive=ec, est_robust=er, alpha=alpha, t_level=tlev, t_coercive=tc, t_robust=tr)
            if ncheck < 12:   # identity checks (not timed): D_r = Löwner D_- ; D_r - D = r^T S^{-1} r
                j = KS.index(K); Nm = LW[j]['Gm'][np.ix_(ix, ix)] + np.diag(1/da)
                Dminus = float(g @ np.linalg.solve(Nm, g))
                Gr = Gam @ r; MGr = st['Mi'] @ Gr
                rSr = float(r @ Gr - MGr @ np.linalg.solve(Gex[np.ix_(ix, ix)] + np.diag(1/da), MGr))
                lev['check_Dr_vs_Dminus'] = abs(Dr - Dminus)/abs(Dminus); lev['check_energy_identity'] = abs((Dr - Dex) - rSr)/max(abs(Dr - Dex), 1e-300)
                lev['true_error'] = Dr - Dex
            item['pr'].append(lev)
        ncheck += 1
        qrec.append(item)
    rec['queries'] = qrec
    if ti == 0:
        for (Rm, da, ar, cl) in [queries[2], queries[4]]:
            t0 = pc(); A = (A0 + D[Rm].T @ sps.diags(da) @ D[Rm]).tocsc(); luA = spla.splu(A); u = luA.solve(f); td = pc()-t0
            out['direct_checks'].append(dict(cl=cl, direct_s=td, D_direct=c0 - float(f @ u), D_woodbury=online_exact(rowmap[Rm], gfull[Rm], da)))
            del luA, A
    del Gex, Gam, LW, luE, G0, M, MZ, W, Z
    out['per_tile'].append(rec); dump()
    print(f"tile {ti+1}/{len(tiles)}: geom {rec['geometry_s']:.1f}s exact {rec['exact_offline_s']:.1f}s shared {t_shared:.1f}s "
          f"Löwner {sum(rec['offline']['loewner_levels_s']):.1f}s extLU {t_ext:.1f}s", flush=True)

# ---------------- summary ----------------
allq = [q for r in out['per_tile'] for q in r['queries']]; nq = len(allq); nt = len(tiles)
offl = dict(A0_LU=t_lu, geometry=sum(r['geometry_s'] for r in out['per_tile']),
            exact=sum(r['exact_offline_s'] for r in out['per_tile']),
            shared=sum(r['offline']['shared_spectral_and_kmax_solves_s'] for r in out['per_tile']),
            loewner_levels=sum(sum(r['offline']['loewner_levels_s']) for r in out['per_tile']),
            pr_exterior=sum(r['offline']['pr_exterior_lu_s'] for r in out['per_tile']))
def method_bounds(q, meth, j):
    if meth == 'loewner':
        x = q['lw'][j]; return x['lo'], x['up']
    x = q['pr'][j]; est = dict(pr_coercive=x['est_coercive'], pr_robust=x['est_robust'], pr_best=min(x['est_coercive'], x['est_robust']))[meth]
    return x['Dr'] - est, x['Dr']
def step_cost(q, meth, j):
    if meth == 'loewner': return q['lw'][j]['t']
    x = q['pr'][j]; c = x['t_level'] + (q['pr_fact_t'] if j == 0 else 0.0)
    if meth in ('pr_coercive', 'pr_best'): c += x['t_coercive']
    if meth in ('pr_robust', 'pr_best'): c += x['t_robust']
    return c
METHODS = ['loewner', 'pr_coercive', 'pr_robust', 'pr_best']
def fixed_cost(q, meth, j):
    if meth == 'loewner': return q['lw'][j]['t']
    x = q['pr'][j]; c = q['pr_fact_t'] + x['t_level']
    if meth in ('pr_coercive', 'pr_best'): c += x['t_coercive']
    if meth in ('pr_robust', 'pr_best'): c += x['t_robust']
    return c
summ = dict(total_queries=nq, offline=offl, exact_online_mean_ms=1e3*np.mean([q['t_exact'] for q in allq]),
            direct_solve_s=np.mean([d['direct_s'] for d in out['direct_checks']]) if out['direct_checks'] else None)
checks = [lv for q in allq for lv in q['pr'] if 'check_Dr_vs_Dminus' in lv]
summ['identity_checks'] = dict(n=len(checks), max_rel_Dr_vs_Dminus=max(c['check_Dr_vs_Dminus'] for c in checks),
                               max_rel_energy_identity=max(c['check_energy_identity'] for c in checks if abs(c['true_error']) > 1e-12*abs(c['Dr'])) if checks else None)
fixed = {}
for meth in METHODS:
    fixed[meth] = {}
    for j, K in enumerate(KS):
        wid = []; cont = 0
        for q in allq:
            lo_, up_ = method_bounds(q, meth, j)
            if np.isfinite(lo_) and np.isfinite(up_):
                wid.append((up_-lo_)/abs(q['D'])); cont += int(lo_ - 1e-12*abs(q['D']) <= q['D'] <= up_ + 1e-12*abs(q['D']))
        per = {}
        for cl, name in enumerate(CLASSES):
            ws_ = [(method_bounds(q, meth, j)[1]-method_bounds(q, meth, j)[0])/abs(q['D']) for q in allq if q['cl'] == cl]
            per[name] = [float(np.nanmedian(ws_)), float(np.nanmax(ws_))]
        fixed[meth][str(K)] = dict(contained=cont, finite=len(wid), width_median=float(np.median(wid)), width_max=float(np.max(wid)),
                                    per_class_width_med_max=per,
                                    online_ms=1e3*float(np.mean([fixed_cost(q, meth, j) for q in allq])))
summ['fixed_level'] = fixed
adapt = {}
for meth in METHODS:
    adapt[meth] = {}
    off = offl['A0_LU'] + offl['geometry'] + offl['shared'] + (offl['loewner_levels'] if meth == 'loewner' else offl['pr_exterior'] if meth in ('pr_robust', 'pr_best') else 0.0)
    for tol in TOLS:
        cert = 0; true_ok = 0; rej = 0; cost = 0.0; lev_used = []; per_cl_rej = {c: 0 for c in CLASSES}
        for q in allq:
            ok = False
            for j, K in enumerate(KS):
                cost += step_cost(q, meth, j); lo_, up_ = method_bounds(q, meth, j)
                if certified(lo_, up_, tol):
                    ok = True; cert += 1; lev_used.append(K)
                    true_ok += int((up_ - lo_) <= tol*abs(q['D']) and lo_ - 1e-12*abs(q['D']) <= q['D'] <= up_ + 1e-12*abs(q['D'])); break
            if not ok: rej += 1; per_cl_rej[CLASSES[q['cl']]] += 1
        adapt[meth][str(tol)] = dict(certified=cert, true_ok=true_ok, rejected=rej, rejected_per_class=per_cl_rej,
                                     levels_used={str(K): lev_used.count(K) for K in KS},
                                     online_total_s=cost, online_mean_ms=1e3*cost/nq, offline_total_s=off,
                                     end_to_end_s=off + cost)
summ['adaptive'] = adapt
summ['exact_end_to_end_s'] = offl['A0_LU'] + offl['geometry'] + offl['exact'] + sum(q['t_exact'] for q in allq)
out['summary'] = summ; dump()
print(json.dumps({k: v for k, v in summ.items() if k != 'fixed_level'}, indent=1))
