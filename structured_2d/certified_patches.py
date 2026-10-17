"""Patch-wise certified compression: positions tiled into t x t tiles; for each tile compress H^grad
restricted to the edges covered by all blocks with position in the tile (exact SVD via QR of row-subsets of X, Y)."""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
exec(open(_os.path.join(HERE, 'certified_online.py')).read().split('if __name__')[0])
pos_of_edge = {ed: k for k, ed in enumerate(edges)}

def tile_offline(Xall, Yall, i_lo, j_lo, t):
    E_p = sorted({e for a in range(i_lo, i_lo+t) for b in range(j_lo, j_lo+t) for e in block_edges(a, b)})
    Qx, Rx = np.linalg.qr(Xall[E_p]); Qy, Ry = np.linalg.qr(Yall[E_p])
    Wm, S, Zt = np.linalg.svd(Rx @ Ry.T)
    return {'E': {e: k for k, e in enumerate(E_p)}, 'U': Qx @ Wm, 'V': Qy @ Zt.T, 'S': S}

m = 2                                   # positions allowed down to 2 cells from boundary
Xall = Dg @ Pk; Yall = Dg @ Kb          # all edges (rows), offline
lo_i, hi_i = m, n-1-m-(s-1)
rng = np.random.default_rng(5)
for t in [8, 16]:
    tiles = {}
    t0 = time.time()
    for a in range(lo_i, hi_i+1, t):
        for b in range(lo_i, hi_i+1, t):
            tiles[(a, b)] = tile_offline(Xall, Yall, a, b, min(t, hi_i+1-a) if False else t if a+t <= hi_i+1 else hi_i+1-a) if False else None
    # build tiles with clipped sizes
    for a in range(lo_i, hi_i+1, t):
        for b in range(lo_i, hi_i+1, t):
            ta, tb = min(t, hi_i+1-a), min(t, hi_i+1-b)
            E_p = sorted({e for x in range(a, a+ta) for y in range(b, b+tb) for e in block_edges(x, y)})
            Qx, Rx = np.linalg.qr(Xall[E_p]); Qy, Ry = np.linalg.qr(Yall[E_p])
            Wm, S, Zt = np.linalg.svd(Rx @ Ry.T)
            tiles[(a, b)] = {'E': {e: k for k, e in enumerate(E_p)}, 'U': Qx @ Wm, 'V': Qy @ Zt.T, 'S': S}
    print(f"\ntile size {t} cells: {len(tiles)} tiles, offline {time.time()-t0:.1f}s")
    for r in [6, 10, 16, 24]:
        eps_all = np.array([T_['S'][r] for T_ in tiles.values()])
        print(f"  r={r:2d}: eps over tiles: median {np.median(eps_all):.1e}, max {eps_all.max():.1e}")
    positions = [(int(a), int(b)) for a, b in rng.integers(lo_i, hi_i+1, size=(30, 2))] + [(lo_i, 40), (lo_i, lo_i), (hi_i, hi_i)]
    for r in [10, 16, 24]:
        nval = ncert = ntot = 0; widths = []
        for (i0, j0) in positions:
            key = (lo_i + ((i0-lo_i)//t)*t, lo_i + ((j0-lo_i)//t)*t)
            T_ = tiles[key]
            loc = [T_['E'][e] for e in block_edges(i0, j0)]
            Hhat = (T_['U'][loc, :r]*T_['S'][:r]) @ T_['V'][loc, :r].T; Hhat = 0.5*(Hhat+Hhat.T)
            Ghat = P0 @ (GK_loc - Hhat) @ P0; eps = T_['S'][r] + 1e-12
            G, g, F = G_and_g(i0, j0)
            assert np.linalg.norm(P0 @ (G-Ghat) @ P0, 2) <= eps*(1+1e-9)
            for case in range(4):
                da = [np.exp(rng.normal(0, 1.5, p))*5, rng.uniform(-0.9, -0.1, p), rng.uniform(-0.5, 5, p), np.full(p, 1e6)][case]
                Dinv = np.diag(1/da); Nm, Np = Dinv + Ghat - eps*P0, Dinv + Ghat + eps*P0
                same = inertia_neg(Nm) == inertia_neg(Np)
                lo, hi = g @ np.linalg.solve(Np, g), g @ np.linalg.solve(Nm, g)
                Dtrue = c0 - f @ spla.spsolve((A0 + F.T @ sps.diags(da) @ F).tocsc(), f)
                ntot += 1
                if same:
                    ncert += 1; nval += (lo - 1e-10*abs(Dtrue) <= Dtrue <= hi + 1e-10*abs(Dtrue)); widths.append((hi-lo)/abs(Dtrue))
        wd = np.array(widths)
        print(f"  r={r:2d}: certified {ncert}/{ntot}, valid {nval}/{ncert}, rel.width median {np.median(wd):.1e} max {wd.max():.1e}")
