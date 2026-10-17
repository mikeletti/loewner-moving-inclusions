"""Randomized and adversarial checks of the main algebraic statements (numbering of the SISC manuscript):
Theorem 3.4 (upper bound without inertia condition) and the need for G_- >= 0; Corollary 3.5 (level zero);
Lemma 4.5 (reduction to range F); Proposition 4.3 / Corollary 5.5 with the projected truncation of Lemma 6.1;
Theorems 5.1 and 5.3 and Remark 5.2(a) with a genuinely singular local Neumann Schur complement;
Lemma 6.3 (one-solve enclosure) and its hypothesis; Lemma 6.8 (spectrum of T, damped rate).

Graph-based generator (weighted incidence matrices with Dirichlet edges only in the exterior), so that the
local Neumann Schur complement S_in is singular for a floating patch (kernel = constants).
Runs in about 10 s. Every printed "worst" value should be at round-off level or satisfy the stated bound.
"""
import numpy as np
rng = np.random.default_rng(7)
worst = {}
count = {}
def upd(k, v): worst[k] = max(worst.get(k, -np.inf), float(v))
def cnt(k, inc=1): count[k] = count.get(k, 0) + inc
def inertia(X, tol=1e-11):
    w = np.linalg.eigvalsh((X + X.T)/2); s = tol*max(1.0, np.abs(w).max())
    return (int((w > s).sum()), int((w < -s).sum()), int((np.abs(w) <= s).sum()))
def psd_sqrt(S, inv=False):
    w, V = np.linalg.eigh((S + S.T)/2); t = 1e-11*max(w.max(), 1e-300); k = w > t
    d = 1/np.sqrt(w[k]) if inv else np.sqrt(w[k]); return (V[:, k]*d) @ V[:, k].T
def orth(X, tol=1e-10):
    U, s, _ = np.linalg.svd(X, full_matrices=False); return U[:, s > tol*max(s.max(), 1e-300)]

def build(touch):
    nI, nB, nE = rng.integers(10, 22), rng.integers(6, 12), rng.integers(10, 20)
    n = nI + nB + nE; I = np.arange(nI); B = nI + np.arange(nB); E = nI + nB + np.arange(nE)
    edges, in_om = [], []
    def add(a, b, om): edges.append((a, b)); in_om.append(om)
    for v in range(1, nI): add(v, rng.integers(0, v), True)                 # spanning tree on I
    for _ in range(nI): a, b = rng.choice(nI, 2, replace=False); add(a, b, True)   # cycles in I
    for v in B: add(v, rng.integers(0, nI), True)                            # each port linked to I
    for _ in range(nB): add(rng.choice(B), rng.integers(0, nI), True)
    for _ in range(nB): a, b = rng.choice(B, 2, replace=False); add(a, b, bool(rng.random() < 0.5))
    for v in E: add(v, rng.choice(np.r_[B, E[E < v]]), False)                # exterior connected to ports
    for _ in range(nE): a, b = rng.choice(np.r_[B, E], 2, replace=False); add(a, b, False)
    for v in rng.choice(E, 3, replace=False): add(v, -1, False)              # Dirichlet edges (exterior)
    if touch:
        for v in rng.choice(B, 2, replace=False): add(v, -1, True)           # patch touching Dirichlet
    m = len(edges); Dg = np.zeros((m, n))
    for e, (a, b) in enumerate(edges):
        Dg[e, a] = 1.0
        if b >= 0: Dg[e, b] = -1.0
    c = rng.uniform(0.5, 2.0, m); in_om = np.array(in_om)
    A0 = Dg.T @ (c[:, None]*Dg); Aom = Dg[in_om].T @ (c[in_om][:, None]*Dg[in_om])
    Aext = Dg[~in_om].T @ (c[~in_om][:, None]*Dg[~in_om])
    rowsI = [e for e, (a, b) in enumerate(edges) if a in I and b in I]
    return dict(n=n, I=I, B=B, E=E, Dg=Dg, c=c, A0=A0, Aom=Aom, Aext=Aext, rowsI=rowsI)

def local(P, rows):
    I, B, A0, Aom = P['I'], P['B'], P['A0'], P['Aom']
    F = P['Dg'][rows]; FI = F[:, I]
    AII = A0[np.ix_(I, I)]; AIB = A0[np.ix_(I, B)]; Ext = -np.linalg.solve(AII, AIB)
    Sin = Aom[np.ix_(B, B)] + Aom[np.ix_(B, I)] @ Ext; Sin = (Sin + Sin.T)/2
    Gam = np.linalg.inv(A0)[np.ix_(B, B)]; Gloc = FI @ np.linalg.solve(AII, FI.T); M = FI @ Ext
    return F, Sin, Gam, Gloc, M

# ---------------------------------------------------------------- A. DN bracketing with singular S_in
for trial in range(200):
    touch = trial % 4 == 3
    P = build(touch); rowsT = P['rowsI']
    F, Sin, Gam, Gloc, M = local(P, rowsT)
    nz = inertia(Sin)[2]
    cnt('A: patches with singular S_in' if nz > 0 else 'A: patches with S_in > 0')
    if nz > 0: upd('A: dim ker S_in (floating patch)', nz)
    Sp = np.linalg.pinv(Sin, rcond=1e-12, hermitian=True); Sh = psd_sqrt(Sin); Shi = psd_sqrt(Sin, inv=True)
    upd('A: (H) first half residual ||M^T - S S^+ M^T||/||M||', np.linalg.norm(M.T - Sin @ Sp @ M.T)/np.linalg.norm(M))
    G = F @ np.linalg.solve(P['A0'], F.T)
    s = np.linalg.svd(M @ Shi, compute_uv=False); kmax = int((s > 1e-10*s[0]).sum())
    k = int(rng.integers(1, max(2, kmax)))
    Vk = np.linalg.svd(M @ Shi)[2][:k].T; W = Sh @ Vk
    Nn = Gam @ W @ np.linalg.solve(W.T @ Gam @ W, W.T @ Gam)
    Pi = Sp @ W @ np.linalg.solve(W.T @ Sp @ W, W.T); T = M @ (np.eye(len(P['B'])) - Pi)
    Gm = Gloc + M @ Nn @ M.T; Gp = Gm + T @ Sp @ T.T
    upd('A: sandwich -lmin(G-G_-)/||G||', -np.linalg.eigvalsh(G - Gm).min()/np.linalg.norm(G, 2))
    upd('A: sandwich -lmin(G_+-G)/||G||', -np.linalg.eigvalsh(Gp - G).min()/np.linalg.norm(G, 2))
    upd('A: | ||G_+-G_-|| - d_k^2 | / d_k^2', abs(np.linalg.norm(Gp - Gm, 2) - s[k]**2)/max(s[k]**2, 1e-300) if s[k] > 1e-8*s[0] else 0)
    # invariance of N and of G_- under W -> W R
    R = rng.normal(size=(k, k)); WR = W @ R
    Nn2 = Gam @ WR @ np.linalg.solve(WR.T @ Gam @ WR, WR.T @ Gam)
    upd('A: ||N(WR)-N(W)||/||N(W)||', np.linalg.norm(Nn2 - Nn)/max(np.linalg.norm(Nn), 1e-300))
    # ranges in range(F) (needed by the width corollary), for one position = random subset of rows
    sub = sorted(rng.choice(len(rowsT), max(2, len(rowsT)//2), replace=False))
    Q = orth(F[sub]); Pmu = Q @ Q.T
    for name, X in (('G_-', Gm), ('G_+', Gp), ('X_r X_r^T', T @ Sp @ T.T)):
        Xm = X[np.ix_(sub, sub)]; upd(f'A: ||(I-P_mu) {name}[mu]||/||{name}[mu]||', np.linalg.norm(Xm - Pmu @ Xm)/max(np.linalg.norm(Xm), 1e-300))

    # ------------------------------------------------------------ B. unconditional D <= D_- (no inertia test)
    lam = P['c'][rowsT]; p = len(rowsT)
    f = rng.normal(size=P['n']); u0 = np.linalg.solve(P['A0'], f); g = F @ u0
    for kind in ('mixed', 'near-void', 'soft', 'harsh'):
        if kind == 'mixed': d = np.where(rng.random(p) < .5, rng.uniform(.1, 20, p), -rng.uniform(.1, .9, p))*lam
        elif kind == 'near-void': d = -rng.uniform(.98, .99999, p)*lam
        elif kind == 'soft': d = -rng.uniform(.1, .9, p)*lam
        else: d = np.where(rng.random(p) < .5, rng.uniform(1, 1e4, p), -rng.uniform(.9, .999, p))*lam
        A = P['A0'] + F.T @ (d[:, None]*F); assert np.linalg.eigvalsh(A).min() > 0
        Dx = g @ np.linalg.solve(np.diag(1/d) + G, g); InD = (int((d > 0).sum()), int((d < 0).sum()), 0)
        cands = [('Nystrom G_-', Gm)]
        Gh = psd_sqrt(G)
        for _ in range(3):   # random 0 <= X <= G, sometimes singular
            Z = rng.normal(size=(p, p)); C = Z @ Z.T; C /= np.linalg.eigvalsh(C).max()
            if rng.random() < .5: C = orth(rng.normal(size=(p, max(1, p//3)))); C = C @ C.T
            cands.append(('random 0<=X<=G', Gh @ C @ Gh))
        for name, X in cands:
            N = np.diag(1/d) + X; In = inertia(N, 1e-13)
            cnt('B: (X, Delta) pairs tested');
            if In != InD: cnt('B: inertia(Delta^-1+X) != In(Delta)')
            sv = np.linalg.svd(N, compute_uv=False).min(); upd('B: -log10 sigma_min(Delta^-1+X) (info)', -np.log10(sv))
            Dm = g @ np.linalg.solve(N, g); upd('B: (D - D_-)/|D| (must be <= 0)', (Dx - Dm)/abs(Dx))
        # hypothesis G_- >= 0 is needed: G_- = G - s I (still <= G) can break the bound
        for sh in (0.5, 2.0, 10.0):
            X = G - sh*np.linalg.norm(G, 2)*np.eye(p); N = np.diag(1/d) + X
            if inertia(N, 1e-13) != InD: cnt('B-: G_- not >= 0: inertia changes (hypothesis needed)')
            Dm = g @ np.linalg.solve(N, g)
            if Dm < Dx*(1 - 1e-12) - 1e-15: cnt('B-: G_- not >= 0: D > D_- (hypothesis needed)')

# ---------------------------------------------------------------- C. size-q reduction and inertia identity
for trial in range(3000):
    p = int(rng.integers(3, 12)); q = int(rng.integers(1, p + 1))
    Q = np.linalg.qr(rng.normal(size=(p, q)))[0]
    d = rng.normal(size=p)*np.exp(rng.normal(size=p)); Dl = np.diag(d)
    Y = rng.normal(size=(q, q)); Y = Y + Y.T
    if trial % 3 == 0: Y = Y @ Y.T                       # PSD, sometimes rank deficient
    if trial % 5 == 0: Y[:, -1] = 0; Y[-1, :] = 0
    Bm = Q.T @ Dl @ Q
    if np.linalg.svd(Bm, compute_uv=False).min() < 1e-6: cnt('C: B singular (skipped)'); continue
    N = np.diag(1/d) + Q @ Y @ Q.T; X = np.linalg.inv(Bm) + Y
    if min(np.linalg.svd(N, compute_uv=False).min(), np.linalg.svd(X, compute_uv=False).min()) < 1e-8: continue
    lhs = inertia(N); InD = inertia(Dl); InB = inertia(Bm); InX = inertia(X)
    rhs = tuple(InD[i] - InB[i] + InX[i] for i in range(3))
    cnt('C: instances');
    if lhs != rhs: cnt('C: In(N) != In(Delta)-In(B)+In(B^-1+Y)')
    k = rng.normal(size=q); g = Q @ k
    upd('C: |g^T N^-1 g - k^T (B^-1+Y)^-1 k| rel', abs(g @ np.linalg.solve(N, g) - k @ np.linalg.solve(X, k))/max(1, abs(g @ np.linalg.solve(N, g))))

# ---------------------------------------------------------------- D. tau*P_mu variant, corollary, one-solve width
for trial in range(300):
    P = build(trial % 3 == 0); rowsT = P['rowsI']
    F, Sin, Gam, Gloc, M = local(P, rowsT)
    Sp = np.linalg.pinv(Sin, rcond=1e-12, hermitian=True); Sh = psd_sqrt(Sin); Shi = psd_sqrt(Sin, inv=True)
    Ms = M @ Shi; Usv = np.linalg.svd(Ms); k = int(rng.integers(1, 4))
    Vk = Usv[2][:k].T; W = Sh @ Vk
    Nn = Gam @ W @ np.linalg.solve(W.T @ Gam @ W, W.T @ Gam); Gm = Gloc + M @ Nn @ M.T
    PV = Vk @ Vk.T; Xr = Ms @ (np.eye(len(P['B'])) - PV)
    sub = sorted(rng.choice(len(rowsT), max(3, (2*len(rowsT))//3), replace=False))
    Fm = F[sub]; Gmu = Fm @ np.linalg.solve(P['A0'], Fm.T); Qm = orth(Fm); Pmu = Qm @ Qm.T
    Xm = Xr[sub]; Gmm = Gm[np.ix_(sub, sub)]
    # truncation with a random (non-optimal) right projection of rank t
    t = int(rng.integers(0, max(1, min(Xm.shape) - 1)))
    Wt = np.linalg.qr(rng.normal(size=(Xr.shape[1], t)))[0] if t > 0 else np.zeros((Xr.shape[1], 0))
    Xt = Xr @ Wt; rho = np.linalg.norm(Xr - Xt @ Wt.T, 'fro')**2
    Rm = Xm @ (np.eye(Xr.shape[1]) - Wt @ Wt.T) @ Xm.T
    upd('D: -lmin(rho P_mu - R[mu])/rho', -np.linalg.eigvalsh(rho*Pmu - Rm).min()/max(rho, 1e-300))
    Ep = Xt[sub] @ Xt[sub].T + rho*Pmu; GpP = Gmm + Ep; eta = np.linalg.norm(Ep, 2)
    upd('D: -lmin(G_+^{t,P} - G_mu)/||G_mu||', -np.linalg.eigvalsh(GpP - Gmu).min()/np.linalg.norm(Gmu, 2))
    wq = np.linalg.eigvalsh(Qm.T @ Gmu @ Qm); gam = wq.min()
    if eta >= gam: cnt('D: eta >= gamma_mu (corollary not applicable)'); continue
    th = eta/(gam - eta)
    lam = P['c'][rowsT][sub]; p = len(sub)
    f = rng.normal(size=P['n']); g = Fm @ np.linalg.solve(P['A0'], f)
    for _ in range(20):
        d = np.exp(rng.uniform(-12, 12, p))*lam                 # stiffening, contrasts 1e-5 .. 1e5
        D = g @ np.linalg.solve(np.diag(1/d) + Gmu, g)
        Nm = np.diag(1/d) + Gmm; w = np.linalg.solve(Nm, g); Dm = g @ w
        Dp = g @ np.linalg.solve(np.diag(1/d) + GpP, g)
        cnt('D: queries (Delta > 0)')
        upd('D: (D_- - D_+)/D / theta  (<= 1)', (Dm - Dp)/D/th)
        upd('D: (w^T E w / D) / (theta(1+theta))  (<= 1)', (w @ Ep @ w)/D/(th*(1 + th)))
        upd('D: one-solve violation (D_- - w^T E w - D_+)/D (<= 0)', (Dm - w @ Ep @ w - Dp)/D)

# ---------------------------------------------------------------- E. Lemma rich (refined spectrum) and Lemma onesolve hypothesis
for trial in range(3000):
    p = int(rng.integers(2, 9)); lam = rng.uniform(.5, 2, p)
    Z = rng.normal(size=(p, p)); G = Z @ Z.T; G = G/np.linalg.eigvalsh(np.diag(np.sqrt(lam)) @ G @ np.diag(np.sqrt(lam))).max()*rng.uniform(.3, .99)  # G <= Lambda^-1
    C = rng.normal(size=(p, p)); C = C @ C.T; C /= np.linalg.eigvalsh(C).max()*rng.uniform(1, 3)
    Gh = psd_sqrt(G); Gm = Gh @ C @ Gh                                   # 0 <= G_- <= G
    Rr = rng.normal(size=(p, 2)); Gp = G + rng.uniform(0, .3)*Rr @ Rr.T/np.linalg.norm(Rr)**2
    Em = Gp - Gm; eta = np.linalg.norm(Em, 2)
    # mixed Delta with beta > 0
    nP = int(rng.integers(0, p + 1)); d = np.empty(p)
    d[:nP] = np.exp(rng.uniform(-2, 6, nP))*lam[:nP]
    hi = 1/(1/lam[nP:] + eta*rng.uniform(1.05, 3, p - nP))              # -1/d - 1/lam > eta
    d[nP:] = -hi*rng.uniform(.05, 1, p - nP)
    alpha = (1/d[:nP]).min() if nP else np.inf
    beta = (-1/d[nP:] - 1/lam[nP:]).min() - eta if nP < p else np.inf
    if not (alpha > 0 and beta > 0): continue
    sig = min(alpha, beta); Nm = np.diag(1/d) + Gm; Np = np.diag(1/d) + Gp
    ev = np.linalg.eigvals(np.linalg.solve(Nm, Em)); cnt('E: instances (Thm apriori hypotheses hold)')
    if eta >= sig: cnt('E: damped regime')
    upd('E: max |Im eig T|', np.abs(ev.imag).max())
    upd('E: -min eig T / (eta/(eta+sigma))  (<= 1)', -ev.real.min()/(eta/(eta + sig)))
    upd('E: max eig T / (eta/sigma)  (<= 1)', ev.real.max()/(eta/sig))
    om = sig/(sig + eta); rhoit = np.abs(1 - om*(1 + ev)).max()
    upd('E: rho(I-om(I+T)) / (1-(sig/(sig+eta))^2)  (<= 1)', rhoit/(1 - om**2))
    if eta < sig: upd('E: ||T||_2 / (eta/sigma)  (<= 1, undamped)', np.linalg.norm(np.linalg.solve(Nm, Em), 2)/(eta/sig))

# Lemma onesolve: without G_- <= G <= G_+ the conclusion fails; with it, it holds
for trial in range(2000):
    p = int(rng.integers(2, 7)); Z = rng.normal(size=(p, p)); G = Z @ Z.T
    d = np.exp(rng.uniform(-3, 3, p)); g = rng.normal(size=p); D = g @ np.linalg.solve(np.diag(1/d) + G, g)
    Gh = psd_sqrt(G); C = rng.normal(size=(p, p)); C = C @ C.T; C /= np.linalg.eigvalsh(C).max()
    Gm = Gh @ C @ Gh; R = rng.normal(size=(p, p)); Em = (G - Gm) + R @ R.T*rng.uniform(0, .5)
    w = np.linalg.solve(np.diag(1/d) + Gm, g); Dm = g @ w; Dp = g @ np.linalg.solve(np.diag(1/d) + Gm + Em, g)
    cnt('F: one-solve with hypothesis'); upd('F: with hyp.: max violation (rel)', max(Dm - w @ Em @ w - Dp, Dp - D, D - Dm)/abs(D))
    # violate the hypothesis: G_- >= 0 and E >= 0 but G not between them
    Gm2 = G + R @ R.T; w2 = np.linalg.solve(np.diag(1/d) + Gm2, g); Dm2 = g @ w2
    if Dm2 < D*(1 - 1e-12): cnt('F-: without hyp.: D > D_- (hypothesis needed)')

# ---------------------------------------------------------------- G. level zero for -a* < delta < 0 (mixed)
for trial in range(1000):
    P = build(False); rows = P['rowsI']; F = P['Dg'][rows]; lam = P['c'][rows]; p = len(rows)
    ast = lam.min()                                    # F^T F <= ast^-1 A0 holds with a* = min c (Lemma diagbound-type)
    G = F @ np.linalg.solve(P['A0'], F.T); u0 = np.linalg.solve(P['A0'], rng.normal(size=P['n'])); g = F @ u0
    d = np.where(rng.random(p) < .5, rng.uniform(.01, 50, p), -rng.uniform(.01, .999, p))*ast
    D = g @ np.linalg.solve(np.diag(1/d) + G, g)
    lo = np.sum(ast*d/(ast + d)*g**2); hi = np.sum(d*g**2)
    cnt('G: level-zero instances (mixed, -a*<delta)')
    upd('G: max violation of level-zero bounds (rel)', max(lo - D, D - hi)/abs(D))

for kk in sorted(count): print(f'{kk:62s} {count[kk]:6d}')
print()
for kk in sorted(worst): print(f'{kk:62s} {worst[kk]: .2e}')
