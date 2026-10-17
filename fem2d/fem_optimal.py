"""Energy-optimal Nystrom enrichment. With S = S_in, choose V_k = top-k right singular vectors of M S^{+1/2},
W = S^{1/2} V_k. Then G_+ - G_- = M S^{+1/2} (I - V_k V_k^T) S^{+1/2} M^T, ||G_+ - G_-|| = d_k^2, d_k = sigma_{k+1}(M S^{+1/2}).
Also: d_k is the Kolmogorov k-width of {grad v|_R : v discrete a-harmonic in omega, a_omega(v,v) <= 1}."""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import sys, numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
mesh_m = int(sys.argv[1]) if len(sys.argv) > 1 else 64
src = open(_os.path.join(HERE, 'fem_bracketing.py')).read().split('if __name__')[0]
src = src.replace('m = 64\n', f'm = {mesh_m}\n')
exec(src)

def tile_pieces(lo, hi, r_inc, ell):
    RT = np.where(dist_to_box(cent, lo, hi) <= r_inc)[0]
    om = set(np.where(dist_to_box(cent, lo, hi) <= r_inc + ell)[0].tolist())
    Vom = set(tri[list(om)].ravel().tolist())
    I = sorted(fid[v] for v in Vom if free[v] and all(e in om for e in inc[v]))
    Iset = set(I)
    Bn = sorted(fid[v] for v in Vom if free[v] and fid[v] not in Iset)
    AII = A0[I][:, I].tocsc(); AIB = A0[I][:, Bn].toarray(); luI = spla.splu(AII)
    E = -luI.solve(AIB)
    S = A0om = assemble(sorted(om), a0)[Bn][:, Bn].toarray() + AIB.T @ E
    S = 0.5*(S + S.T)
    Din = grad_rows(RT)[:, I].toarray()
    G0 = Din @ luI.solve(Din.T); Mm = Din @ E
    ws, Vs = np.linalg.eigh(S); keep = ws > 1e-10*ws.max()
    Sh = (Vs[:, keep]*np.sqrt(ws[keep])) @ Vs[:, keep].T          # S^{1/2}
    Shp = (Vs[:, keep]/np.sqrt(ws[keep])) @ Vs[:, keep].T         # S^{+1/2}
    return dict(RT=RT, Bn=Bn, G0=G0, M=Mm, S=S, Sh=Sh, Shp=Shp, nB=len(Bn), nI=len(I), kerS=int((~keep).sum()))

def enclosure(T_, k, mode):
    Mm, Bn = T_['M'], T_['Bn']
    if mode == 'energy':
        _, d, Vt = np.linalg.svd(Mm @ T_['Shp'], full_matrices=False)
        Vk = Vt[:k].T; W = T_['Sh'] @ Vk
    else:
        _, d, Vt = np.linalg.svd(Mm, full_matrices=False); W = Vt[:k].T
    if k > 0:
        rhs = np.zeros((Nf, k)); rhs[Bn, :] = W
        GW = lu0.solve(rhs)[Bn, :]
        MG = Mm @ GW; Nm = MG @ np.linalg.solve(W.T @ GW, MG.T)
    else:
        Nm = 0*T_['G0']
    Gm = T_['G0'] + Nm
    Sp = T_['Shp'] @ T_['Shp']
    if mode == 'energy':
        X = Mm @ T_['Shp']; Xr = X - (X @ Vk) @ Vk.T if k > 0 else X
        Gp = Gm + Xr @ Xr.T
    else:
        Tm = Mm - (Mm @ W) @ W.T if k > 0 else Mm
        Gp = Gm + Tm @ Sp @ Tm.T
    return 0.5*(Gm+Gm.T), 0.5*(Gp+Gp.T)

if __name__ == "__main__":
    r_inc, ell = 0.06, 0.08
    lo, hi = np.array([0.45, 0.45]), np.array([0.55, 0.55])
    T_ = tile_pieces(lo, hi, r_inc, ell)
    _, d, _ = np.linalg.svd(T_['M'] @ T_['Shp'], full_matrices=False)
    print(f"mesh {mesh_m}: N_free={Nf}, tile |I|={T_['nI']}, |B|={T_['nB']}, dim ker S_in={T_['kerS']}")
    print("  d_k = sigma_{k+1}(M S^{+1/2}) / d_0 for k = 0,2,4,8,12,16,24,32:",
          " ".join(f"{v:.1e}" for v in d[[0, 2, 4, 8, 12, 16, 24, 32]]/d[0]), f"  (d_0 = {d[0]:.3f})")
    # exact tile G for checks
    F_T = grad_rows(T_['RT']); Gex = F_T @ lu0.solve(F_T.T.toarray()); Gex = 0.5*(Gex+Gex.T)
    for mode in ['euclid', 'energy']:
        for k in [4, 8, 16]:
            Gm, Gp = enclosure(T_, k, mode)
            lo_ok = np.linalg.eigvalsh(Gex-Gm).min() > -1e-9; hi_ok = np.linalg.eigvalsh(Gp-Gex).min() > -1e-9
            gap = np.linalg.norm(Gp-Gm, 2)
            extra = f"  (d_k^2 = {d[k]**2:.2e})" if mode == 'energy' else ""
            print(f"  {mode:6s} k={k:2d}: sandwich ok {lo_ok and hi_ok}, ||G_+ - G_-|| = {gap:.2e}{extra}")
