"""Online core of the single-factorization Loewner evaluation (note v0.5.3).

qd_upper_bound: certified lower bound on D_+ = g^T N_+^{-1} g (i.e. an upper bound on J = c0 - D)
when N_- and N_+ are quasi-definite with sigma_min >= sigma (Theorem 'a priori inertia').
Iteration: x <- x + omega N_-^{-1} (g - N_+ x), with omega = 1 if eta < sigma, else 1/(1 + eta/sigma).
  * eigenvalues of T = N_-^{-1}(N_+ - N_-) are real, lie in (-1, eta/sigma] and in [-eta/sigma, eta/sigma];
  * hence the undamped iteration contracts with factor <= eta/sigma when eta < sigma, and the damped one
    has iteration-matrix spectrum in [0, 1) otherwise (convergent, possibly slowly).
The returned bound is valid for ANY iterate (Proposition 'a posteriori enclosure'); convergence only
affects its width. If the tolerance is not reached within maxit steps, the caller falls back to a
second factorization.
Exact-arithmetic statements; floating-point computations are not validated.
"""
import numpy as np

def qd_upper_bound(solve_minus, matvec_plus, g, x0, sigma, eta, tol=1e-13, maxit=60):
    omega = 1.0 if eta < sigma else 1.0/(1.0 + eta/sigma)
    x = x0.copy(); L = -np.inf; res = np.inf
    for it in range(maxit + 1):
        r = g - matvec_plus(x)
        res = float(r @ r)/sigma
        L = float(g @ x + x @ r) - res            # certified: g^T N_+^{-1} g >= L
        if res <= tol*abs(float(g @ x)):
            return L, it, res, omega, True
        if it < maxit:
            x = x + omega*solve_minus(r)
    return L, maxit, res, omega, False
