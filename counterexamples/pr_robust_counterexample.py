"""Theorem 7.1 (port-reduced Galerkin = D_-), energy identity, by-product, and the three certificates
(Loewner / PR robust / PR coercive), including globally-but-not-pointwise admissible Delta."""
import numpy as np
from common import *

rng = np.random.default_rng(36)
m = 18; n = m * m
Dg, E = grad_op(m)
ends = [(a, b) for (a, b, _) in E]


def box_edges(i0, j0, s):
    out = []
    for k, (a, b) in enumerate(ends):
        if b < 0: continue
        ia, ja = divmod(a, m); ib, jb = divmod(b, m)
        if all(i0 <= x < i0 + s for x in (ia, ib)) and all(j0 <= y < j0 + s for y in (ja, jb)):
            out.append(k)
    return out


def setup(lam, om_box, incl_box):
    A0 = Dg.T @ (lam[:, None] * Dg)
    om = set(box_edges(*om_box))
    inc = {x: [] for x in range(n)}
    for k, (a, b) in enumerate(ends):
        inc[a].append(k)
        if b >= 0: inc[b].append(k)
    nodes = set()
    for k in om:
        nodes.update([x for x in ends[k] if x >= 0])
    I = sorted(x for x in nodes if all(k in om for k in inc[x]))
    B = sorted(nodes - set(I)); O = sorted(set(range(n)) - set(I) - set(B))
    rows = box_edges(*incl_box)
    for k in rows: assert all(x in I for x in ends[k])
    rest = sorted(set(range(len(E))) - om)
    Aext = Dg[rest].T @ (lam[rest][:, None] * Dg[rest])
    Sext = Aext[np.ix_(B, B)] - Aext[np.ix_(B, O)] @ np.linalg.solve(Aext[np.ix_(O, O)], Aext[np.ix_(O, B)])
    return A0, I, B, O, rows, Sext


def run(lam, d, rows_sel, A0, I, B, O, rows, Sext, k, f):
    F = Dg[rows_sel]; FI = F[:, I]
    A = A0 + F.T @ (d[:, None] * F)
    assert lmin(A) > 0
    A0inv = np.linalg.inv(A0)
    Gam = A0inv[np.ix_(B, B)]
    AII = A0[np.ix_(I, I)]; AIB = A0[np.ix_(I, B)]
    Ext = -np.linalg.solve(AII, AIB)
    M = FI @ Ext; Gloc = FI @ np.linalg.solve(AII, FI.T)
    u0 = A0inv @ f; g = F @ u0; c0 = f @ u0
    W = rng.standard_normal((len(B), k))
    K = W.T @ Gam @ W
    Gm = Gloc + M @ Gam @ W @ np.linalg.solve(K, W.T @ Gam @ M.T)
    Nm = np.diag(1 / d) + Gm
    Dminus = g @ np.linalg.solve(Nm, g)
    # direct port-reduced Galerkin: v(z) with exact perturbed interior / reference exterior
    AdII = A[np.ix_(I, I)]
    def lift(z):
        v = np.zeros(n); v[B] = z
        v[I] = np.linalg.solve(AdII, f[I] - A[np.ix_(I, B)] @ z)
        v[O] = np.linalg.solve(A[np.ix_(O, O)], f[O] - A[np.ix_(O, B)] @ z)
        return v
    z0 = u0[B]
    v0 = lift(z0)
    V = np.column_stack([lift(z0 + Gam @ W[:, j]) - v0 for j in range(k)])
    # maximise J(v0+Vy) = 2f'(v0+Vy) - (v0+Vy)'A(v0+Vy)
    H = V.T @ A @ V; q = V.T @ (f - A @ v0)
    y = np.linalg.solve(H, q); ur = v0 + V @ y
    Jr = 2 * f @ ur - ur @ A @ ur
    u = np.linalg.solve(A, f); J = f @ u
    # port Schur complement S(Delta), residual
    S = A[np.ix_(B, B)] - A[np.ix_(B, I)] @ np.linalg.solve(AdII, AIB) - A[np.ix_(B, O)] @ np.linalg.solve(A[np.ix_(O, O)], A[np.ix_(O, B)])
    bD = f[B] - A[np.ix_(B, I)] @ np.linalg.solve(AdII, f[I]) - A[np.ix_(B, O)] @ np.linalg.solve(A[np.ix_(O, O)], f[O])
    zr = ur[B]; r = bD - S @ zr
    Er = Dminus - (c0 - J)
    out = dict(id1=abs((c0 - Jr) - Dminus) / abs(Dminus), id2=abs((J - Jr) - (u - ur) @ A @ (u - ur)) / abs(J - Jr),
               id3=abs((J - Jr) - r @ np.linalg.solve(S, r)) / abs(J - Jr), fur_vs_Jr=abs(f @ ur - Jr) / abs(Jr),
               Er=Er, robust=r @ np.linalg.solve(Sext, r), Spatch_min=lmin(S - Sext),
               Nm_sing=np.linalg.svd(Nm, compute_uv=False)[-1], In_ok=inertia(Nm) == inertia(np.diag(d)))
    alpha = min(1.0, np.min(1 + d / lam[rows_sel]))
    out['alpha'] = alpha
    out['coercive'] = r @ Gam @ r / alpha if alpha > 0 else np.nan
    out['rr_form'] = np.linalg.norm(r - (-M.T @ np.linalg.solve(np.diag(1 / d) + Gloc, g + M @ Gam @ W @ y) - W @ y)) / np.linalg.norm(r)
    return out


lam = rng.uniform(0.5, 2.0, len(E))
A0, I, B, O, rows, Sext = setup(lam, (3, 3, 12), (6, 6, 6))
print(f"setting: |I|={len(I)} |B|={len(B)} |O|={len(O)} rows of D_in={len(rows)}; lmin(S_ext)={lmin(Sext):.2e}")
worst = dict(id1=0, id2=0, id3=0, rr_form=0); minfr = 1; robust_fail = 0; coer_fail = 0; tot = 0; inert_bad = 0
for it in range(120):
    sub = sorted(rng.choice(rows, size=int(rng.integers(3, len(rows))), replace=False))  # row-subset setting
    p = len(sub); L = lam[sub]
    cls = it % 3
    if cls == 0: d = L * 10 ** rng.uniform(-2, 4, p)
    elif cls == 1: d = -L * (1 - 10 ** rng.uniform(-4, -0.2, p))
    else: d = np.where(rng.random(p) < 0.5, -L * rng.uniform(0.1, 0.999, p), L * 10 ** rng.uniform(-1, 3, p))
    f = rng.standard_normal(n)
    o = run(lam, d, sub, A0, I, B, O, rows, Sext, int(rng.integers(1, 10)), f)
    tot += 1
    for kk in worst: worst[kk] = max(worst[kk], o[kk])
    minfr = min(minfr, o['fur_vs_Jr'])
    if o['Er'] > o['robust'] * (1 + 1e-8): robust_fail += 1
    if o['Er'] > o['coercive'] * (1 + 1e-8): coer_fail += 1
    if not o['In_ok']: inert_bad += 1
print(f"Theorem 7.1 (pointwise admissible, {tot} queries, random row subsets, k in 1..9): max rel errors "
      f"c0-J_r vs D_-: {worst['id1']:.1e}; energy identity: {worst['id2']:.1e}; r'S^-1 r: {worst['id3']:.1e}; "
      f"residual formula r=-M'N0^-1(g+Ly)-Wy: {worst['rr_form']:.1e}; min |f'u_r-J_r|/|J_r| = {minfr:.1e}; "
      f"In(N_-)=In(Delta) failures {inert_bad}; PR robust failures {robust_fail}; PR coercive failures {coer_fail}")

# ---- globally admissible but not pointwise admissible Delta: one edge with negative total conductance ----
kneg = rows[len(rows) // 2]
sub = [kneg] + [k for k in rows if k != kneg][:6]
a_, b_ = ends[kneg]
# effective conductance across edge kneg in A0 minus that edge (global) and in the patch only
def eff_cond(Amat, a, b, nodes):
    idx = {x: i for i, x in enumerate(nodes)}
    Ar = Amat[np.ix_(nodes, nodes)]
    e = np.zeros(len(nodes)); e[idx[a]] = 1; e[idx[b]] = -1
    x = np.linalg.lstsq(Ar, e, rcond=None)[0]
    return 1 / (e @ x)
Arest = A0 - lam[kneg] * np.outer(Dg[kneg], Dg[kneg])
cg = eff_cond(Arest, a_, b_, list(range(n)))
oml = box_edges(3, 3, 12); oml = [k for k in oml if k != kneg]
Apatch = Dg[oml].T @ (lam[oml][:, None] * Dg[oml])
cp = eff_cond(Apatch, a_, b_, sorted(set(I) | set(B)))
print(f"edge {kneg}: effective conductance of the rest: global {cg:.3f}, patch only (Neumann on B) {cp:.3f}")
cneg = -(cp + cg) / 2  # total edge conductance lam+delta, negative, between the two thresholds
fails = 0; tot = 0; ex = None
for it in range(200):
    d = np.concatenate([[cneg - lam[kneg]], lam[sub[1:]] * rng.uniform(0.2, 3, len(sub) - 1)])
    f = rng.standard_normal(n)
    o = run(lam, d, sub, A0, I, B, O, rows, Sext, int(rng.integers(1, 6)), f)
    tot += 1
    if o['Er'] > o['robust'] * (1 + 1e-8):
        fails += 1
        if ex is None or o['Er'] / o['robust'] > ex[0]: ex = (o['Er'] / o['robust'], o)
print(f"global-only admissible Delta (lam+delta = {cneg:.3f} < 0 on one edge, A(Delta) SPD): "
      f"lmin(S_patch(Delta)) = {o['Spatch_min']:.3e}; alpha = {o['alpha']:.3f}; Theorem 7.1 identity err {o['id1']:.1e}; "
      f"PR robust certificate violated in {fails}/{tot} queries" + (f" (worst E_r / bound = {ex[0]:.2f})" if ex else ""))
