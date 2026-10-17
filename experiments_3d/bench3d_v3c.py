"""3D single-tile offline/online comparison: exact tile Woodbury (|B| global solves) vs Loewner (k global solves).
Shared local geometry (patch LU, harmonic extension E, S_in, G0, M) is timed separately and counted for both.
Single-threaded BLAS recommended. Usage: python bench3d_v3c.py n [k] [trunc_rank]"""
import sys, json, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, scipy.linalg as sla
from time import perf_counter as pc
n = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 16; TT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
h = 1.0/(n+1); N = n**3; idx = lambda i, j, k: (i*n + j)*n + k
t = pc()
rows, cols, vals, mid, en = [], [], [], [], []; e = 0
for ax in range(3):
    for i in range(-1, n):
        for j in range(n):
            for k in range(n):
                p = [j, k]; p.insert(ax, i); q = list(p); q[ax] += 1; nodes = []
                for s_, node in ((-1.0, p), (1.0, q)):
                    if 0 <= node[ax] < n:
                        rows.append(e); cols.append(idx(*node)); vals.append(s_/h); nodes.append(idx(*node))
                    else: nodes.append(-1)
                mid.append([(c + 1 + (0.5 if d == ax else 0))*h for d, c in enumerate(p)]); en.append(nodes); e += 1
D = sps.csr_matrix((vals, (rows, cols)), shape=(e, N)); mid = np.array(mid); en = np.array(en)
a0 = 1.0 + 0.4*np.sin(3*mid[:, 0])*np.cos(2*mid[:, 1]) + 0.3*mid[:, 2]
A0 = (D.T @ sps.diags(a0) @ D).tocsc()
t_asm = pc()-t
t = pc(); lu0 = spla.splu(A0); t_lu = pc()-t
x1 = np.zeros(N); x1[N//2] = 1; t = pc(); [lu0.solve(x1) for _ in range(5)]; t_solve1 = (pc()-t)/5
X = np.array([[(i+1)*h, (j+1)*h, (k+1)*h] for i in range(n) for j in range(n) for k in range(n)])
f = np.exp(-20*((X[:, 0]-0.35)**2 + (X[:, 1]-0.6)**2 + (X[:, 2]-0.5)**2)) + 0.3
u0 = lu0.solve(f); c0 = f @ u0
inc = [[] for _ in range(N)]
for ed, (a_, b_) in enumerate(en):
    if a_ >= 0: inc[a_].append(ed)
    if b_ >= 0: inc[b_].append(ed)
def dbox(c, lo, hi):
    dd = np.maximum(np.maximum(lo-c, 0), np.maximum(c-hi, 0)); return np.linalg.norm(dd, axis=1)
r_inc, ell = 0.10, 0.10
lo, hi = np.array([0.45]*3), np.array([0.55]*3)
# ---- shared local geometry
t = pc()
RT = np.where(dbox(mid, lo, hi) <= r_inc)[0]
om = np.where(dbox(mid, lo, hi) <= r_inc+ell)[0]; oms = set(om.tolist())
Vom = set(en[om].ravel().tolist()) - {-1}
I = np.array(sorted(v for v in Vom if all(x in oms for x in inc[v]))); Is = set(I.tolist())
B = np.array(sorted(v for v in Vom if v not in Is))
AII = A0[I][:, I].tocsc(); AIB = A0[I][:, B].toarray(); luI = spla.splu(AII)
E = -luI.solve(AIB)
Dom = D[om]; Aom = (Dom.T @ sps.diags(a0[om]) @ Dom).tocsr()
S = Aom[B][:, B].toarray() + AIB.T @ E; S = 0.5*(S+S.T)
Din = D[RT][:, I].tocsr(); DinT = Din.T.tocsc(); G0 = np.empty((len(RT), len(RT)))
for c_ in range(0, len(RT), 1024):                     # sparse D_in, chunked local solves
    cs = slice(c_, min(c_+1024, len(RT))); G0[:, cs] = Din @ luI.solve(DinT[:, cs].toarray())
G0 = 0.5*(G0+G0.T); M = np.asarray(Din @ E)
t_geom = pc()-t
del E
# ---- exact tile Woodbury: Gamma = A0^{-1}[B,B] via |B| global solves
t = pc(); rhs = np.zeros((N, len(B))); rhs[B, np.arange(len(B))] = 1.0
Gam = lu0.solve(rhs)[B, :]; del rhs; t_gam = pc()-t
t = pc(); Gex = G0 + M @ Gam @ M.T; Gex = 0.5*(Gex+Gex.T); t_exasm = pc()-t
# ---- Loewner energy-optimal enrichment (randomized SVD of Z = M S^{+1/2})
t = pc(); ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max()
Sh = (Vs[:, keep]*np.sqrt(ws[keep])) @ Vs[:, keep].T; Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T; t_eig = pc()-t
t = pc(); Z = M @ Shp
rng = np.random.default_rng(0); Om = rng.normal(size=(Z.shape[1], K+10)); Y = Z @ Om
for _ in range(2): Y = Z @ (Z.T @ Y)
Qr, _ = np.linalg.qr(Y); _, sv, Vt = np.linalg.svd(Qr.T @ Z, full_matrices=False); Vk = Vt[:K].T
t_rsvd = pc()-t
t = pc(); W = Sh @ Vk; rhs = np.zeros((N, K)); rhs[B, :] = W; GW = lu0.solve(rhs)[B, :]
MG = M @ GW; Gm = G0 + MG @ np.linalg.solve(W.T @ GW, MG.T); Gm = 0.5*(Gm+Gm.T)
Xr = Z - (Z @ Vk) @ Vk.T                      # gap factor: G_+ - G_- = Xr Xr^T
t_nys = pc()-t
t = pc()
Om2 = rng.normal(size=(Xr.shape[0], TT+10)); Y2 = Xr.T @ Om2
for _ in range(2): Y2 = Xr.T @ (Xr @ Y2)
Wt, _ = np.linalg.qr(Y2); Wt = Wt[:, :TT+10]          # right subspace (|B| x (TT+10)), orthonormal
Xt = Xr @ Wt                                           # rows x (TT+10)
Rres = Xr - Xt @ Wt.T; rho = float(np.sum(Rres*Rres)); del Rres          # ||Xr (I - Wt Wt^T)||_F^2 computed directly (no cancellation)
tau = rho*(1 + 1e-8) + 1e-15*float(np.sum(Xr*Xr))                        # small relative inflation + floor (not a verified roundoff bound)
t_trunc = pc()-t
# ---- certified-sandwich sanity (exact arithmetic statement; fp check) on the whole tile
gap_norm = float(sv[K]**2) if len(sv) > K else 0.0
lam_lo = lam_hi = None
# ---- online comparison on positions in the tile
xt_norm2 = float(np.linalg.norm(Xt, 2)**2)                 # offline: ||Xt||_2^2 (enters the a-priori inertia test)
rowmap = {ed: j for j, ed in enumerate(RT)}
tests = []; rr = np.random.default_rng(5)
for _ in range(4):
    mu = lo + rr.uniform(0, 1, 3)*(hi-lo); Rm = np.where(np.linalg.norm(mid-mu, axis=1) <= r_inc)[0]
    ix = np.array([rowmap[x] for x in Rm]); g = D[Rm] @ u0; ar = a0[Rm]
    for cl, da in enumerate([np.exp(rr.normal(0, 1.5, len(Rm)))*5, rr.uniform(-0.9, -0.1, len(Rm))*ar, rr.uniform(-0.5, 5, len(Rm))*ar, np.full(len(Rm), 1e6)]):
        tests.append((ix, g, da, ar, cl))
from scipy.linalg import lapack
_lw = {}
def sytrf(Nm):                                           # blocked Bunch-Kaufman: optimal lwork (default lwork=n is unblocked!)
    p = Nm.shape[0]
    if p not in _lw: _lw[p] = int(lapack.dsytrf_lwork(p, lower=1)[0])
    lu, ipiv, info = lapack.dsytrf(Nm, lower=1, lwork=_lw[p], overwrite_a=1); return lu, ipiv
def sytrs(f, rhs): return lapack.dsytrs(f[0], f[1], rhs, lower=1)[0]
def neg_inertia(f):
    lu, ipiv = f; d1 = ipiv > 0; neg = int(np.sum(np.diag(lu)[d1] < 0)); st = np.where(ipiv < 0)[0][::2]
    if len(st):
        a_, b_, c_ = lu[st, st], lu[st+1, st], lu[st+1, st+1]; det = a_*c_ - b_*b_
        neg += int(np.sum(det < 0) + 2*np.sum((det >= 0) & (a_ + c_ < 0)))
    return neg
def online_exact(ix, g, da, ar, cl):                     # same factorization routine as Loewner (fair)
    f = sytrf(Gex[np.ix_(ix, ix)] + np.diag(1/da)); return c0 - g @ sytrs(f, g)
stats = []
def online_loewner(ix, g, da, ar, cl):
    Nm = Gm[np.ix_(ix, ix)] + np.diag(1/da); Xi = Xt[ix]
    if np.all(da > 0):                                   # Cholesky + one-solve enclosure with truncated gap Xt Xt^T + tau I
        c, info = lapack.dpotrf(Nm, lower=1); w = lapack.dpotrs(c, g, lower=1)[0]
        Dm = g @ w; v = Xi.T @ w; stats.append(('chol', 0, 0.0)); return (c0 - Dm, c0 - Dm + v @ v + tau*(w @ w))
    P, Q = da > 0, da < 0
    alpha = float(np.min(1/da[P])) if P.any() else np.inf
    beta = float(np.min(-1/da[Q] - 1/ar[Q])) - tau - xt_norm2 if Q.any() else np.inf
    if min(alpha, beta) > 0:                             # a-priori inertia: N_- and N_+ quasi-definite, In = (|P|,|Q|)
        sig = min(alpha, beta)                           # sigma_min(N_+) >= min(alpha, beta)
        f = sytrf(Nm.copy()); x = sytrs(f, g); Jlo = c0 - g @ x
        for it in range(40):                             # Richardson on N_+ = N_- + Xi Xi^T + tau I, preconditioned by N_-
            r = g - (Nm @ x + Xi @ (Xi.T @ x) + tau*x)
            L = g @ x + x @ r - (r @ r)/sig              # certified: g^T N_+^{-1} g >= L
            if (r @ r)/sig <= 1e-13*abs(g @ x): break
            x = x + sytrs(f, r)
        stats.append(('qd', it, (r @ r)/sig/abs(g @ x))); return (Jlo, c0 - L)
    Np = Nm + Xi @ Xi.T + tau*np.eye(len(ix))            # fallback: two factorizations + inertia check
    fm = sytrf(Nm.copy()); fp = sytrf(Np)
    stats.append(('2fact', 0, 0.0))
    if neg_inertia(fm) != neg_inertia(fp): return (np.nan, np.nan)
    return (c0 - g @ sytrs(fm, g), c0 - g @ sytrs(fp, g))
def online_loewner_2fact(ix, g, da, ar, cl):            # v2 algorithm with blocked sytrf, for comparison
    Nm = Gm[np.ix_(ix, ix)] + np.diag(1/da); Xi = Xt[ix]
    if np.all(da > 0): return online_loewner(ix, g, da, ar, cl)
    Np = Nm + Xi @ Xi.T + tau*np.eye(len(ix)); fm = sytrf(Nm.copy()); fp = sytrf(Np)
    if neg_inertia(fm) != neg_inertia(fp): return (np.nan, np.nan)
    return (c0 - g @ sytrs(fm, g), c0 - g @ sytrs(fp, g))
def sandwich_check(ix):
    Ge = Gex[np.ix_(ix, ix)]; Gl = Gm[np.ix_(ix, ix)]; Xi = Xt[ix]
    return (np.linalg.eigvalsh(Ge - Gl).min(), np.linalg.eigvalsh(Gl + Xi @ Xi.T + tau*np.eye(len(ix)) - Ge).min())
def timed(fn):
    res, tim = [], []
    for tt in tests:
        t = pc(); res.append(fn(*tt)); tim.append(pc()-t)
    return res, np.array(tim)
online_exact(*tests[0]); online_loewner(*tests[1]); stats.clear()     # warm-up (lwork query, BLAS threads)
ex, tex = timed(online_exact); lw, tlw = timed(online_loewner); qd_keep = list(stats); lw2, tlw2 = timed(online_loewner_2fact); stats[:] = qd_keep
t_on_ex, t_on_lw, t_on_lw2 = tex.mean(), tlw.mean(), tlw2.mean()
cls = np.array([tt[4] for tt in tests]); per_class = {str(c): [1e3*tex[cls == c].mean(), 1e3*tlw[cls == c].mean(), 1e3*tlw2[cls == c].mean()] for c in range(4)}
chk = [sandwich_check(tt[0]) for tt in tests[::4]]; lam_lo = min(c[0] for c in chk); lam_hi = min(c[1] for c in chk)
ok = [l - 1e-9*abs(x) <= x <= u + 1e-9*abs(x) for (l, u), x in zip(lw, ex) if np.isfinite(l)]
ok2 = [l - 1e-9*abs(x) <= x <= u + 1e-9*abs(x) for (l, u), x in zip(lw2, ex) if np.isfinite(l)]
wid = [(u-l)/abs(x) for (l, u), x in zip(lw, ex) if np.isfinite(l)]
Dex = np.array([c0 - x for x in ex]); Wabs = np.array([u - l for (l, u) in lw])
Dup = np.array([c0 - l for (l, u) in lw]); Dlo = np.array([c0 - u for (l, u) in lw])   # D_- (upper on D), D_+ (lower on D)
mlo = (Dex - Dlo)/np.abs(Dex); mup = (Dup - Dex)/np.abs(Dex)                          # signed margins relative to |D| (>=0 means contained)
okD = [bool(a >= -1e-12 and b >= -1e-12) for a, b in zip(mlo, mup)]
margins = {str(c): [float(np.min(mlo[cls == c])), float(np.min(mup[cls == c]))] for c in range(4)}

widD = Wabs/np.abs(Dex); widJ = Wabs/np.abs(np.array(ex))
per_class_D = {str(c): [float(np.median(widD[cls == c])), float(np.max(widD[cls == c]))] for c in range(4)}
Dratio = {str(c): float(np.median(np.abs(Dex[cls == c])/abs(c0))) for c in range(4)}
out = dict(n=n, N=N, k=K, nB=int(len(B)), nI=int(len(I)), rows=int(len(RT)), p_mean=float(np.mean([len(tt[0]) for tt in tests])),
           A0_LU_s=t_lu, one_global_solve_s=t_solve1, shared_local_geometry_s=t_geom,
           exact_gamma_solves_s=t_gam, exact_assembly_s=t_exasm, exact_total_s=t_gam+t_exasm,
           loewner_eigh_s=t_eig, loewner_rsvd_s=t_rsvd, loewner_nystrom_s=t_nys, loewner_trunc_s=t_trunc, loewner_total_s=t_eig+t_rsvd+t_nys+t_trunc, trunc_rank=TT+10, tau=tau,
           gap_norm=gap_norm, sandwich_min_eigs=[lam_lo, lam_hi],
           online_ms_exact=1e3*t_on_ex, online_ms_loewner=1e3*t_on_lw, online_ms_loewner_2fact=1e3*t_on_lw2, online_ms_per_class_exact_qd_2fact=per_class,
           qd_stats=[(int(s_[1]), float(s_[2])) for s_ in stats if s_[0] == 'qd'], valid_2fact=int(sum(ok2)), xt_norm2=xt_norm2,
           certified=f"{len(ok)}/{len(tests)}", valid=int(sum(ok)), valid_strict_D=int(sum(okD)), min_signed_margin_over_D_per_class_lower_upper=margins, rho=rho, relwidth_median=float(np.median(wid)), relwidth_max=float(np.max(wid)), relwidthJ_median=float(np.median(widJ)), relwidthD_median=float(np.median(widD)), relwidthD_max=float(np.max(widD)), relwidthD_per_class_med_max=per_class_D, D_over_c0_per_class=Dratio, c0=float(c0),
           storage_MB_exact=Gex.nbytes/1e6, storage_MB_loewner_factored=(Gm.nbytes+Xr.nbytes)/1e6, storage_MB_loewner_truncated=(Gm.nbytes+Xt.nbytes)/1e6)
try:
    import resource; out["peak_RSS_GB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
except ImportError:
    try:
        import psutil; out["peak_RSS_GB"]=psutil.Process().memory_info().peak_wset/1e9
    except Exception: out["peak_RSS_GB"]=None
print(json.dumps(out, indent=1))
json.dump(out, open(f'bench3d_v3c_n{n}_k{K}_r{TT}.json', 'w'), indent=1)
