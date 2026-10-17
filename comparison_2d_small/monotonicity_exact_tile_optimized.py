"""Optimized exact-tile Woodbury baseline: no local Schur eigendecomposition or SVD.
Run from any folder with BENCH_M=32/48/64.
Global A0 LU common setup is separately timed. Independent of monotonicity_enrichment.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import os,sys,json,time
import numpy as np
from time import perf_counter
import scipy.sparse.linalg as sla
MESH_M=int(os.environ.get('BENCH_M','64'))
sys.argv=['fem_bracketing.py']
src=open(_os.path.join(HERE, '..', 'fem2d', 'fem_bracketing.py')).read().split('if __name__')[0].replace('m = 64\n',f'm = {MESH_M}\n')
t0=perf_counter();exec(compile(src,'fem_bracketing.py','exec'));setup=perf_counter()-t0
side=[0.30,0.40,0.50];tiles=[(np.array([a,b]),np.array([a+.10,b+.10])) for a in side for b in side]
count=0; total_rt=0; total_b=0; result=[]
t0=perf_counter()
for lo,hi in tiles:
    RT=np.where(dist_to_box(cent,lo,hi)<=.06)[0]
    om=set(np.where(dist_to_box(cent,lo,hi)<=.14)[0].tolist())
    Vom=set(tri[list(om)].ravel().tolist()); Iset={fid[v] for v in Vom if free[v] and all(e in om for e in inc[v])}
    I=np.array(sorted(Iset),int); B=np.array(sorted(fid[v] for v in Vom if free[v] and fid[v] not in Iset),int)
    AII=A0[I][:,I].tocsc(); AIB=A0[I][:,B].toarray(); lui=sla.splu(AII)
    E=-lui.solve(AIB); Din=grad_rows(RT)[:,I].toarray()
    G0=Din@lui.solve(Din.T); Mm=Din@E
    rhs=np.zeros((Nf,len(B))); rhs[B,np.arange(len(B))]=1
    Gamma=lu0.solve(rhs)[B,:]
    Gt=G0+Mm@Gamma@Mm.T
    count+=len(B);total_rt+=2*len(RT);total_b+=len(B)
    result.append(Gt)
t=perf_counter()-t0
out=dict(mesh_m=MESH_M,n_free=Nf,tiles=len(tiles),setup_s=setup,optimized_exact_tile_offline_s=t,total_B_RHS=count,mean_nB=total_b/len(tiles),mean_tile_G_size=total_rt/len(tiles),storage_MB=sum(g.nbytes for g in result)/1e6)
print(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
open(f'monotonicity_exact_tile_opt_m{MESH_M}.json','w').write(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
