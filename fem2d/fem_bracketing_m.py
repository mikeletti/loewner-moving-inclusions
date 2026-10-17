"""Mesh-agnostic certified enclosure: unstructured P1 FEM, variable reference coefficient.
For a tile of positions with inner region R_T (union of inclusions) and patch omega (collar l):
  A0^{-1}[I,I] = A_II^{-1} + E Gamma E^T,  Gamma = A0^{-1}[B,B] (exact DD identity)
  Gamma = N + C,  N = Gamma W (W^T Gamma W)^{-1} W^T Gamma (Nystrom, k global solves), C >= 0, C W = 0
  0 <= T C T^T <= T S_in^+ T^T   (S_in = interior Neumann Schur complement, local)
=> G_- = G0 + M N M^T <= G <= G_- + T S_in^+ T^T = G_+, with M = D_in E, T = M (I - W W^T).
"""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
from scipy.spatial import Delaunay
rng = np.random.default_rng(0)

# ---------------- mesh ----------------
import sys; m = int(sys.argv[1])
g1 = np.linspace(0, 1, m+1)
X, Y = np.meshgrid(g1, g1, indexing='ij')
pts = np.c_[X.ravel(), Y.ravel()]
onb = (np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 1) | np.isclose(pts[:, 1], 0) | np.isclose(pts[:, 1], 1))
pts[~onb] += rng.uniform(-0.3, 0.3, size=((~onb).sum(), 2))/m
tri = Delaunay(pts).simplices
P = pts[tri]
area = 0.5*np.abs((P[:, 1, 0]-P[:, 0, 0])*(P[:, 2, 1]-P[:, 0, 1]) - (P[:, 2, 0]-P[:, 0, 0])*(P[:, 1, 1]-P[:, 0, 1]))
keep = area > 1e-12; tri = tri[keep]; P = P[keep]; area = area[keep]
cent = P.mean(1)
# P1 gradients: B_T (2x3)
def grads(Pt):
    M_ = np.c_[np.ones(3), Pt]
    C_ = np.linalg.inv(M_)
    return C_[1:, :]
Bt = np.array([grads(Pt) for Pt in P])           # nE x 2 x 3
a0 = 1.0 + 0.5*np.sin(3*cent[:, 0])*np.cos(2*cent[:, 1]) + 0.3*cent[:, 1]   # variable reference coefficient
free = ~onb
fid = -np.ones(len(pts), int); fid[free] = np.arange(free.sum()); Nf = free.sum()
nE = len(tri)
print(f"mesh: {len(pts)} nodes, {Nf} free, {nE} triangles, a0 in [{a0.min():.2f},{a0.max():.2f}]")

def assemble(elems, coef):
    r, c, v = [], [], []
    for e in elems:
        K = coef[e]*area[e]*Bt[e].T @ Bt[e]
        for a in range(3):
            ia = fid[tri[e, a]]
            if ia < 0: continue
            for b in range(3):
                ib = fid[tri[e, b]]
                if ib < 0: continue
                r.append(ia); c.append(ib); v.append(K[a, b])
    return sps.csr_matrix((v, (r, c)), shape=(Nf, Nf))

A0 = assemble(range(nE), a0).tocsc(); lu0 = spla.splu(A0)
xf = pts[free]
f = np.exp(-30*((xf[:, 0]-0.3)**2 + (xf[:, 1]-0.6)**2)) + 0.3
u0 = lu0.solve(f); c0 = f @ u0

def grad_rows(elems):
    """F: rows sqrt(area) B_T (2 per element), columns = free nodes"""
    r, c, v = [], [], []
    for k, e in enumerate(elems):
        for d in range(2):
            for a in range(3):
                ia = fid[tri[e, a]]
                if ia >= 0:
                    r.append(2*k+d); c.append(ia); v.append(np.sqrt(area[e])*Bt[e][d, a])
    return sps.csr_matrix((v, (r, c)), shape=(2*len(elems), Nf))

# node -> incident elements
inc = [[] for _ in range(len(pts))]
for e, t in enumerate(tri):
    for a in t: inc[a].append(e)

def dist_to_box(c, lo, hi):
    d = np.maximum(np.maximum(lo - c, 0), np.maximum(c - hi, 0))
    return np.hypot(d[:, 0], d[:, 1])

def build_tile(lo, hi, r_inc, ell, k, count):
    RT = np.where(dist_to_box(cent, lo, hi) <= r_inc)[0]
    om = set(np.where(dist_to_box(cent, lo, hi) <= r_inc + ell)[0].tolist())
    Vom = set(tri[list(om)].ravel().tolist())
    I = sorted(fid[v] for v in Vom if free[v] and all(e in om for e in inc[v]))
    Bn = sorted(fid[v] for v in Vom if free[v] and fid[v] not in set(I))
    Iset = set(I)
    assert all(fid[v] in Iset for v in tri[RT].ravel() if free[v]), "collar too thin"
    AII = A0[I][:, I].tocsc(); AIB = A0[I][:, Bn].toarray()
    luI = spla.splu(AII)
    E = -luI.solve(AIB)                                   # harmonic extension
    Aom = assemble(sorted(om), a0)
    S_in = Aom[Bn][:, Bn].toarray() + AIB.T @ E           # A^om_BB - A_BI A_II^{-1} A_IB
    S_in = 0.5*(S_in + S_in.T)
    Din = grad_rows(RT)[:, I].toarray()
    G0 = Din @ luI.solve(Din.T)
    Mm = Din @ E
    _, sv, Vt = np.linalg.svd(Mm, full_matrices=False)
    W = Vt[:k].T
    if k > 0:
        rhs = np.zeros((Nf, k)); rhs[Bn, :] = W
        Z = lu0.solve(rhs); count[0] += k                  # k global solves
        GW = Z[Bn, :]
        Nmat = (Mm @ GW) @ np.linalg.solve(W.T @ GW, (Mm @ GW).T)
        Tm = Mm - (Mm @ W) @ W.T
    else:
        Nmat = 0*G0; Tm = Mm
    ws, Vs = np.linalg.eigh(S_in); keepS = ws > 1e-10*ws.max()
    Spinv = (Vs[:, keepS]/ws[keepS]) @ Vs[:, keepS].T
    Gm = G0 + Nmat; Gp = Gm + Tm @ Spinv @ Tm.T
    rowmap = {e: j for j, e in enumerate(RT)}
    return dict(RT=RT, rowmap=rowmap, Gm=0.5*(Gm+Gm.T), Gp=0.5*(Gp+Gp.T), Din=Din, I=I, sv=sv,
                nI=len(I), nB=len(Bn), kerS=int((~keepS).sum()))

def inertia_neg(Mx): return int((np.linalg.eigvalsh(0.5*(Mx+Mx.T)) < 0).sum())

if __name__ == "__main__":
    r_inc = 0.06
    print(f"inclusion = elements with centroid within r={r_inc} of mu; tiles of side 0.1 in mu in [0.15,0.85]^2")
    for ell in [0.08]:
        for k in [0, 8, 16]:
            count = [0]; t0 = time.time()
            tiles = {}
            edges_ = np.arange(0.15, 0.85+1e-9, 0.1)
            for a in edges_[:-1]:
                for b in edges_[:-1]:
                    lo, hi = np.array([a, b]), np.array([a+0.1, b+0.1])
                    tiles[(round(a, 2), round(b, 2))] = (lo, hi, build_tile(lo, hi, r_inc, ell, k, count))
            t_off = time.time()-t0
            nval = ncert = ntot = 0; widths = []; lowner_ok = True; ps = []
            for (key, (lo, hi, T_)) in list(tiles.items()):
                for _ in range(3):
                    mu = lo + rng.uniform(0, 1, 2)*(hi-lo)
                    Rm = np.where(np.hypot(*(cent - mu).T) <= r_inc)[0]
                    rows = np.array([[2*T_['rowmap'][e], 2*T_['rowmap'][e]+1] for e in Rm]).ravel()
                    Gm, Gp = T_['Gm'][np.ix_(rows, rows)], T_['Gp'][np.ix_(rows, rows)]
                    F = grad_rows(Rm)
                    Gex = (F @ lu0.solve(F.T.toarray())); Gex = 0.5*(Gex+Gex.T)
                    lowner_ok &= np.linalg.eigvalsh(Gex - Gm).min() > -1e-10 and np.linalg.eigvalsh(Gp - Gex).min() > -1e-10
                    g = F @ u0; p_ = len(rows); ps.append(p_)
                    for case in range(4):
                        da_e = [np.exp(rng.normal(0, 1.5, len(Rm)))*5, rng.uniform(-0.9, -0.1, len(Rm))*a0[Rm],
                                rng.uniform(-0.5, 5, len(Rm))*a0[Rm], np.full(len(Rm), 1e6)][case]
                        da = np.repeat(da_e, 2); Dinv = np.diag(1/da)
                        ntot += 1
                        if inertia_neg(Dinv+Gm) != inertia_neg(Dinv+Gp): continue
                        ncert += 1
                        lo_b, hi_b = g @ np.linalg.solve(Dinv+Gp, g), g @ np.linalg.solve(Dinv+Gm, g)
                        Dw = g @ np.linalg.solve(Dinv + Gex, g)     # Woodbury reference
                        nval += (lo_b - 1e-10*abs(Dw) <= Dw <= hi_b + 1e-10*abs(Dw))
                        widths.append((hi_b - lo_b)/abs(Dw))
            wd = np.array(widths)
            print(f"collar {ell:.2f}, k={k:2d}: global solves {count[0]:4d} (N_free={Nf}), offline {t_off:5.1f}s | Loewner sandwich verified: {lowner_ok} | "
                  f"certified {ncert}/{ntot}, valid {nval}/{ncert}, rel.width median {np.median(wd):.1e} max {wd.max():.1e} | p in [{min(ps)},{max(ps)}]")
