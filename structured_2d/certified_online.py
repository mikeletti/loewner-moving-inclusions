import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
exec(open(_os.path.join(HERE, 'certified_offline.py')).read().split('# ---------- 2)')[0])
d = np.load(_os.path.join(HERE, 'offline_data.npz')); Pk, Kb = d['Pk'], d['Kb']
pos_of_edge = {ed: k for k, ed in enumerate(edges)}

# reference local objects (translation invariant)
i_ref = j_ref = 43
eid_ref = block_edges(i_ref, j_ref)
nodes_ref = [idx(i_ref+a, j_ref+b) for a in range(s) for b in range(s)]
Floc = Dg[eid_ref, :][:, nodes_ref].toarray()                    # p x s^2
bi, bj = np.divmod(np.arange(s*s), s)
GammaK = h*h*K(bi[:, None]-bi[None, :], bj[:, None]-bj[None, :])  # translation-invariant part
GK_loc = Floc @ GammaK @ Floc.T
Gref = Floc @ lu.solve(np.eye(N)[:, nodes_ref])[nodes_ref, :] @ Floc.T
w, V = np.linalg.eigh(Gref); Q0 = V[:, w > 1e-10*w.max()]; P0 = Q0 @ Q0.T

def offline(m):
    """compress H^grad on edges with both endpoints at distance >= m cells from the ghost layer"""
    E_d = [k for k, ed in enumerate(edges)
           if (ed[0] == 'x' and m <= ed[1] and ed[1]+1 <= n-1-m and m <= ed[2] <= n-1-m) or
              (ed[0] == 'y' and m <= ed[1] <= n-1-m and m <= ed[2] and ed[2]+1 <= n-1-m)]
    Dd = Dg[E_d, :]
    X = Dd @ Pk; Y = Dd @ Kb
    Qx, Rx = np.linalg.qr(X); Qy, Ry = np.linalg.qr(Y)
    Wm, S, Zt = np.linalg.svd(Rx @ Ry.T)
    U = Qx @ Wm; Vv = Qy @ Zt.T
    return {'E': {e: k for k, e in enumerate(E_d)}, 'U': U, 'V': Vv, 'S': S, 'm': m}

def online(off, i0, j0, r, Gref_norm=1.0):
    loc = [off['E'][e] for e in block_edges(i0, j0)]
    Hhat = (off['U'][loc, :r]*off['S'][:r]) @ off['V'][loc, :r].T
    Hhat = 0.5*(Hhat + Hhat.T)
    Ghat = P0 @ (GK_loc - Hhat) @ P0
    eps = off['S'][r] + 1e-12*Gref_norm        # certified tail + floating-point floor
    return Ghat, eps

def inertia_neg(M):
    return int((np.linalg.eigvalsh(0.5*(M+M.T)) < 0).sum())

if __name__ == "__main__":
    for m in [6, 2]:
        t0 = time.time(); off = offline(m)
        print(f"\nOmega_delta = {m} cells from boundary: |E_delta| = {len(off['E'])}, offline compression {time.time()-t0:.1f}s")
        print("  sigma_j(H^grad)/||G||, j=1..:", " ".join(f"{v:.0e}" for v in off['S'][[0, 4, 9, 14, 19, 29, 39, 59]]), " (j=1,5,10,15,20,30,40,60)")
        rng = np.random.default_rng(3)
        lo_i, hi_i = m, n-1-m-(s-1)
        positions = [(int(a), int(b)) for a, b in rng.integers(lo_i, hi_i+1, size=(25, 2))] + [(lo_i, 40), (lo_i, lo_i)]
        for r in [10, 20, 30, 40]:
            nval = ncert = ntot = 0; widths = []; t_on = 0.0
            for (i0, j0) in positions:
                G, g, F = G_and_g(i0, j0)
                # consistency: exact G must lie in the certified enclosure
                Ghat, eps = online(off, i0, j0, r)
                assert np.linalg.norm(P0 @ (G - Ghat) @ P0, 2) <= eps*(1+1e-9), "enclosure violated"
                for case in range(4):
                    da = [np.exp(rng.normal(0, 1.5, p))*5, rng.uniform(-0.9, -0.1, p), rng.uniform(-0.5, 5, p), np.full(p, 1e6)][case]
                    Dinv = np.diag(1/da)
                    t1 = time.time()
                    Gm, Gp = Ghat - eps*P0, Ghat + eps*P0
                    Nm, Np = Dinv + Gm, Dinv + Gp
                    same = inertia_neg(Nm) == inertia_neg(Np)
                    lo, hi = g @ np.linalg.solve(Np, g), g @ np.linalg.solve(Nm, g)
                    t_on += time.time()-t1
                    Dtrue = c0 - f @ spla.spsolve((A0 + F.T @ sps.diags(da) @ F).tocsc(), f)
                    ntot += 1
                    if same:
                        ncert += 1
                        tol = 1e-10*abs(Dtrue)
                        nval += (lo - tol <= Dtrue <= hi + tol)
                        widths.append((hi-lo)/abs(Dtrue))
            wd = np.array(widths) if widths else np.array([np.nan])
            print(f"  r={r:2d}: eps={off['S'][r]:.1e}  certified {ncert}/{ntot}, valid among certified {nval}/{ncert}, "
                  f"rel.width median {np.median(wd):.1e} max {wd.max():.1e}, online time/eval {1e3*t_on/ntot:.2f} ms")
