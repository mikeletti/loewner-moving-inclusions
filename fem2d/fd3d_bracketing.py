"""3D test of the mesh-agnostic bracketing (Theorem 5.1 + energy-optimal enrichment).
7-point FD on the unit cube, Dirichlet, variable edge coefficient a0; 'elements' = edges.
A0 = D^T diag(a0_e) D; inclusion = edges with midpoint within r of mu; F = rows of D."""
import sys, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
mode_full = (len(sys.argv) <= 2)
h = 1.0/(n+1); N = n**3
idx = lambda i, j, k: (i*n + j)*n + k
rows, cols, vals, mid, enodes = [], [], [], [], []
e = 0
for ax in range(3):
    for i in range(-1, n):
        for j in range(n):
            for k in range(n):
                a = [i, j, k]; b = [i, j, k]; a_ = list(a); b_ = list(b)
                # edge between p and p+e_ax, along axis ax; permute indices
                p = [j, k]; p.insert(ax, i); q = list(p); q[ax] += 1
                nodes = []
                for s_, node in ((-1.0, p), (1.0, q)):
                    if 0 <= node[ax] < n:
                        rows.append(e); cols.append(idx(*node)); vals.append(s_/h); nodes.append(idx(*node))
                    else:
                        nodes.append(-1)
                mid.append([(c + 1 + (0.5 if d == ax else 0))*h for d, c in enumerate(p)])
                enodes.append(nodes); e += 1
D = sps.csr_matrix((vals, (rows, cols)), shape=(e, N)); mid = np.array(mid); enodes = np.array(enodes)
a0 = 1.0 + 0.4*np.sin(3*mid[:, 0])*np.cos(2*mid[:, 1]) + 0.3*mid[:, 2]
A0 = (D.T @ sps.diags(a0) @ D).tocsc()
t0 = time.time(); lu0 = spla.splu(A0); print(f"3D grid n={n}: N={N}, edges={e}, a0 in [{a0.min():.2f},{a0.max():.2f}], global LU {time.time()-t0:.1f}s")
X = np.array([[(i+1)*h, (j+1)*h, (k+1)*h] for i in range(n) for j in range(n) for k in range(n)])
f = np.exp(-20*((X[:, 0]-0.35)**2 + (X[:, 1]-0.6)**2 + (X[:, 2]-0.5)**2)) + 0.3
u0 = lu0.solve(f); c0 = f @ u0
inc = [[] for _ in range(N)]
for ed, (p_, q_) in enumerate(enodes):
    if p_ >= 0: inc[p_].append(ed)
    if q_ >= 0: inc[q_].append(ed)

def dist_to_box(c, lo, hi):
    dd = np.maximum(np.maximum(lo - c, 0), np.maximum(c - hi, 0)); return np.linalg.norm(dd, axis=1)

def tile(lo, hi, r_inc, ell):
    RT = np.where(dist_to_box(mid, lo, hi) <= r_inc)[0]
    om = np.where(dist_to_box(mid, lo, hi) <= r_inc + ell)[0]; oms = set(om.tolist())
    Vom = set(enodes[om].ravel().tolist()) - {-1}
    I = sorted(v for v in Vom if all(ed in oms for ed in inc[v])); Iset = set(I)
    Bn = sorted(v for v in Vom if v not in Iset)
    assert all(v in Iset for v in enodes[RT].ravel() if v >= 0)
    AII = A0[I][:, I].tocsc(); AIB = A0[I][:, Bn].toarray(); luI = spla.splu(AII)
    E = -luI.solve(AIB)
    Dom = D[om]; Aom = (Dom.T @ sps.diags(a0[om]) @ Dom).tocsr()
    S = Aom[Bn][:, Bn].toarray() + AIB.T @ E; S = 0.5*(S+S.T)
    Din = D[RT][:, I].toarray()
    G0 = Din @ luI.solve(Din.T); Mm = Din @ E
    ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max()
    Sh = (Vs[:, keep]*np.sqrt(ws[keep])) @ Vs[:, keep].T; Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T
    Xm = Mm @ Shp; _, d, Vt = np.linalg.svd(Xm, full_matrices=False)
    return dict(RT=RT, rowmap={ed: j for j, ed in enumerate(RT)}, Bn=Bn, G0=G0, Xm=Xm, Mm=Mm, Sh=Sh, Vt=Vt, d=d, nI=len(I), nB=len(Bn))

def enclosure(T_, k):
    Vk = T_['Vt'][:k].T
    if k > 0:
        W = T_['Sh'] @ Vk; rhs = np.zeros((N, k)); rhs[T_['Bn'], :] = W
        GW = lu0.solve(rhs)[T_['Bn'], :]; MG = T_['Mm'] @ GW
        Gm = T_['G0'] + MG @ np.linalg.solve(W.T @ GW, MG.T)
        Xr = T_['Xm'] - (T_['Xm'] @ Vk) @ Vk.T
    else:
        Gm = T_['G0'].copy(); Xr = T_['Xm']
    Gp = Gm + Xr @ Xr.T
    return 0.5*(Gm+Gm.T), 0.5*(Gp+Gp.T)

def inertia_neg(Mx): return int((np.linalg.eigvalsh(0.5*(Mx+Mx.T)) < 0).sum())

r_inc, ell, ts = 0.12, 0.10, 0.1
rng = np.random.default_rng(2)
if not mode_full:
    T_ = tile(np.array([0.45]*3), np.array([0.55]*3), r_inc, ell)
    print(f"  tile |I|={T_['nI']} |B|={T_['nB']}; d_k/d_0, k=0,4,8,16,24,32,48: " + " ".join(f"{v:.1e}" for v in T_['d'][[0, 4, 8, 16, 24, 32, 48]]/T_['d'][0]) + f"  d_0={T_['d'][0]:.3f}")
    sys.exit()
tiles = []
t0 = time.time()
for a in np.arange(0.4, 0.6-1e-9, ts):
    for b in np.arange(0.4, 0.6-1e-9, ts):
        for c in np.arange(0.4, 0.6-1e-9, ts):
            lo = np.array([a, b, c]); tiles.append((lo, lo+ts, tile(lo, lo+ts, r_inc, ell)))
print(f"{len(tiles)} tiles, local offline {time.time()-t0:.1f}s; tile |I| ~ {tiles[0][2]['nI']}, |B| ~ {tiles[0][2]['nB']}")
T0 = tiles[0][2]
print("  d_k/d_0 (first tile), k=0,4,8,16,24,32,48:", " ".join(f"{v:.1e}" for v in T0['d'][[0, 4, 8, 16, 24, 32, 48]]/T0['d'][0]))
tests = []
for (lo, hi, T_) in tiles:
    for _ in range(2):
        mu = lo + rng.uniform(0, 1, 3)*(hi-lo)
        Rm = np.where(np.linalg.norm(mid - mu, axis=1) <= r_inc)[0]
        rws = np.array([T_['rowmap'][ed] for ed in Rm])
        F = D[Rm]; Gex = F @ lu0.solve(F.T.toarray()); Gex = 0.5*(Gex+Gex.T)
        das = [np.exp(rng.normal(0, 1.5, len(Rm)))*5, rng.uniform(-0.9, -0.1, len(Rm))*a0[Rm],
               rng.uniform(-0.5, 5, len(Rm))*a0[Rm], np.full(len(Rm), 1e6)]
        tests.append((T_, rws, Gex, F @ u0, das))
print(f"{len(tests)} positions, p in [{min(len(t[1]) for t in tests)},{max(len(t[1]) for t in tests)}]")
for k in [0, 8, 16, 32]:
    encl = {id(T_): enclosure(T_, k) for (_, _, T_) in tiles}
    ok_all = True; ncert = nval = ntot = 0; widths = []
    for (T_, rws, Gex, g, das) in tests:
        Gm, Gp = (Mx[np.ix_(rws, rws)] for Mx in encl[id(T_)])
        ok_all &= np.linalg.eigvalsh(Gex-Gm).min() > -1e-9 and np.linalg.eigvalsh(Gp-Gex).min() > -1e-9
        for da in das:
            Dinv = np.diag(1/da); ntot += 1
            if inertia_neg(Dinv+Gm) != inertia_neg(Dinv+Gp): continue
            ncert += 1
            lo_b, hi_b = g @ np.linalg.solve(Dinv+Gp, g), g @ np.linalg.solve(Dinv+Gm, g)
            Dw = g @ np.linalg.solve(Dinv+Gex, g)
            nval += (lo_b - 1e-10*abs(Dw) <= Dw <= hi_b + 1e-10*abs(Dw)); widths.append((hi_b-lo_b)/abs(Dw))
    wd = np.array(widths)
    print(f"  k={k:2d}: global solves {k*len(tiles):4d} (N={N}) | sandwich ok {ok_all} | certified {ncert}/{ntot}, valid {nval}/{ncert}, rel.width median {np.median(wd):.1e} max {wd.max():.1e}")
