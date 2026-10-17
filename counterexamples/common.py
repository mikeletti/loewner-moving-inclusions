"""Common helpers for the independent referee tests (v0.7.2 note)."""
import numpy as np
import scipy.linalg as sla


def grid_edges(m):
    """m x m free nodes, ghost Dirichlet layer. Returns list of edges (a,b) with
    a,b node indices or -1 for ghost; node index = i*m+j."""
    E = []
    for i in range(m):
        for j in range(m):
            a = i * m + j
            # right edge
            E.append((a, a + 1 if j + 1 < m else -1, (i, j, 'h')))
            # up edge
            E.append((a, a + m if i + 1 < m else -1, (i, j, 'v')))
            if j == 0:
                E.append((a, -1, (i, j, 'hl')))
            if i == 0:
                E.append((a, -1, (i, j, 'vl')))
    return E


def grad_op(m, E=None):
    if E is None:
        E = grid_edges(m)
    n = m * m
    D = np.zeros((len(E), n))
    for k, (a, b, _) in enumerate(E):
        D[k, a] = 1.0
        if b >= 0:
            D[k, b] = -1.0
    return D, E


def inertia(M, tol=None):
    w = np.linalg.eigvalsh((M + M.T) / 2)
    if tol is None:
        tol = 1e-12 * max(1.0, np.max(np.abs(w)))
    return (int(np.sum(w > tol)), int(np.sum(np.abs(w) <= tol)), int(np.sum(w < -tol)))


def psd_sqrt(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def pinv_sqrt(A, rtol=1e-10):
    w, V = np.linalg.eigh((A + A.T) / 2)
    t = rtol * max(w.max(), 1e-300)
    ws = np.where(w > t, 1 / np.sqrt(np.where(w > t, w, 1)), 0.0)
    return (V * ws) @ V.T


def range_basis(A, rtol=1e-10):
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > rtol * s[0])) if s.size and s[0] > 0 else 0
    return U[:, :r]


def lmin(A):
    return np.linalg.eigvalsh((A + A.T) / 2)[0]


def lmax(A):
    return np.linalg.eigvalsh((A + A.T) / 2)[-1]
