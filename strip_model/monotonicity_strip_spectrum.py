"""Exact modal spectrum of discrete-harmonic gradient restriction on a 2D periodic strip.

Model: m-periodic x, y=0,...,m, five-point Laplace harmonic extension
from data on y=0 and y=m. Energy sums vertical and tangential edge squares.
Inner patch: y grid rows d,...,m-d (d=ceil(ell*m)).
It is an independently reproducible MODEL CHECK, not the general FEM theorem.
"""
import numpy as np
from scipy.linalg import eigh

def spectrum(m, ell=0.15):
    d=int(np.ceil(m*ell)); total=[]; modes=[]
    for k in range(m):
        angle=2*np.pi*k/m
        lamb=4*np.sin(angle/2)**2
        kap=np.arccosh(1+lamb/2)
        y=np.arange(m+1)
        if k==0:
            A=1-y/m; B=y/m
        else:
            # Stable hyperbolic ratio without overflow
            def ratio(a):
                return np.exp(-kap*(m-a))*(-np.expm1(-2*kap*a))/(-np.expm1(-2*kap*m))
            A=ratio(m-y); B=ratio(y)
        V=np.vstack([A,B]).T
        def energy(start,stop):
            vert=np.diff(V[start:stop+1],axis=0)
            return vert.T@vert + lamb*V[start:stop+1].T@V[start:stop+1]
        full=energy(0,m); inner=energy(d,m-d)
        # k=0 null constant, set known nonzero eigenvalue for linear profile
        if k==0:
            vals=[(m-2*d)/m]
        else:
            vals=np.clip(eigh(inner,full,eigvals_only=True),0,1)
        for val in vals:
            total.append(np.sqrt(val)); modes.append((k,np.sqrt(val)))
    vals=np.array(sorted(total,reverse=True))
    return vals, modes, d
if __name__=='__main__':
    for m in [32,64,128]:
        vals,modes,d=spectrum(m)
        print(f'm={m:3d}, collar={d/m:.4f}, n_modes={len(vals)}; d_k at k=0,4,8,16,24,32,48,64:',
             ' '.join(f'{vals[k]:.4e}' for k in [0,4,8,16,24,32,48,64] if k < len(vals)))
        print(' k-mode attenuation roots for k=1,2,4,8:', ' '.join(f'{max(x for q,x in modes if q==k):.3e}' for k in [1,2,4,8]))
