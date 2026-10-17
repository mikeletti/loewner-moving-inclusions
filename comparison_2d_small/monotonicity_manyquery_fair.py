"""Independent comparative many-query pilot for monotonicity_bounds.zip.
Run from any folder: BENCH_M=32 BENCH_REPS=12 python comparison_2d_small/monotonicity_manyquery_fair.py
Uses original FEM model without modifying source. Includes 9 moving-inclusion tiles,
54 held-out positions, 648 queries, multiple methods and offline/online timings.
All methods use the same pre-materialized query geometries and loads.
Times exclude common mesh assembly and the (common) A0 LU setup, but are reported separately.

This is a research pilot, NOT a hardware-neutral benchmark or interval-arithmetic certificate.
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))

import os, time, sys, json
MESH_M=int(os.environ.get('BENCH_M','32'))
NREPS=int(os.environ.get('BENCH_REPS','12'))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import eigh
from time import perf_counter
sys.argv=['fem_bracketing.py']
src=open(_os.path.join(HERE, '..', 'fem2d', 'fem_bracketing.py'),encoding='utf8').read().split('if __name__')[0].replace('m = 64\n',f'm = {MESH_M}\n')
tcommon=perf_counter(); exec(compile(src,'fem_bracketing.py','exec')); tcommon=perf_counter()-tcommon
rng=np.random.default_rng(2601016)
SIDE=[0.30,0.40,0.50]; ELL=0.08; RINC=0.06; K=12
Tiles={ (i,j):(np.array([a,b]),np.array([a+0.10,b+0.10])) for i,a in enumerate(SIDE) for j,b in enumerate(SIDE)}

# Choose 6 independent positions per tile, and 12 coefficient profiles per position.
positions=[]
for tid,(lo,hi) in Tiles.items():
    for posno in range(6):
        mu=lo+(0.1+0.8*rng.random(2))*(hi-lo)
        Rm=np.where(np.linalg.norm(cent-mu,axis=1)<=RINC)[0]
        F=grad_rows(Rm).tocsr(); positions.append(dict(tile=tid,pos=posno,mu=mu,Rm=Rm,F=F,g=np.asarray(F@u0).ravel()))
queries=[]
for rep in range(NREPS):
    for pi,pos in enumerate(positions):
        Rm=pos['Rm']; t=rep%4
        if t==0: de=np.exp(rng.normal(0,1.0,len(Rm)))*3.
        elif t==1: de=rng.uniform(-0.90,-0.1,len(Rm))*a0[Rm]
        elif t==2: de=rng.uniform(-0.5,5,len(Rm))*a0[Rm]
        else: de=np.full(len(Rm),1e6)
        queries.append(dict(pos_id=pi,da=np.repeat(de,2),ctype=['stiff','soft','mixed','rigid'][t]))

# Common patch geometry, with energy-optimal trial subspace.
def geometry(tile):
    lo,hi=Tiles[tile]
    RT=np.where(dist_to_box(cent,lo,hi)<=RINC)[0]
    om=set(np.where(dist_to_box(cent,lo,hi)<=RINC+ELL)[0].tolist())
    Vom=set(tri[list(om)].ravel().tolist()); Iset={fid[v] for v in Vom if free[v] and all(e in om for e in inc[v])}
    I=np.array(sorted(Iset),int); B=np.array(sorted(fid[v] for v in Vom if free[v] and fid[v] not in Iset),int)
    luI=sla.splu(A0[I][:,I].tocsc()); AIB=A0[I][:,B].toarray()
    E=-luI.solve(AIB)
    Sin=assemble(sorted(om),a0)[B][:,B].toarray()+AIB.T@E; Sin=(Sin+Sin.T)/2
    vals,V=eigh(Sin); keep=vals>1e-10*vals[-1]
    sqrtS=(V[:,keep]*np.sqrt(vals[keep]))@V[:,keep].T
    invsqrtS=(V[:,keep]/np.sqrt(vals[keep]))@V[:,keep].T
    D=grad_rows(RT)[:,I].toarray(); M=D@E; G0=D@luI.solve(D.T)
    Z=M@invsqrtS
    _,dd,Vt=np.linalg.svd(Z,full_matrices=False)
    Vtop=Vt[:K].T
    W=sqrtS@Vtop
    rowmap={int(e):i for i,e in enumerate(RT)}
    return dict(B=B,G0=(G0+G0.T)/2,M=M,Z=Z,W=W,Vtop=Vtop,dd=dd,rowmap=rowmap,rt=len(RT),nB=len(B))

t0=perf_counter(); patches={k:geometry(k) for k in Tiles}; tgeom=perf_counter()-t0
# Energy-optimal low-rank certified enclosures, independently per tile.
t0=perf_counter(); count_lo=0
for ti,P in patches.items():
    B=P['B']; W=P['W']; M=P['M']; Z=P['Z']; Vtop=P['Vtop']
    rhs=np.zeros((Nf,K));rhs[B]=W
    GW=lu0.solve(rhs)[B]; count_lo+=K
    MG=M@GW
    Nmat=MG@np.linalg.solve(W.T@GW,MG.T)
    zres=Z-(Z@Vtop)@Vtop.T
    P['Rfactor']=zres
    P['Gminus']=P['G0']+Nmat;P['Gplus']=P['Gminus']+zres@zres.T
    P['Gminus']=(P['Gminus']+P['Gminus'].T)/2
    P['Gplus']=(P['Gplus']+P['Gplus'].T)/2
trank=perf_counter()-t0
# Exact tile Schur baseline -- exact Gamma via ALL interface RHS; may be more costly offline.
t0=perf_counter(); count_ex=0
for P in patches.values():
    B=P['B']; rhs=np.zeros((Nf,len(B)));rhs[B,np.arange(len(B))]=1
    GBB=lu0.solve(rhs)[B];count_ex+=len(B)
    P['Gexact']=P['G0']+P['M']@GBB@P['M'].T
    P['Gexact']=(P['Gexact']+P['Gexact'].T)/2
texact=perf_counter()-t0

# Map positions to tile submatrices once (same metadata available for all algorithms).
t0=perf_counter()
for P in positions:
    patch=patches[P['tile']]
    rr=np.array([row for e in P['Rm'] for row in [2*patch['rowmap'][int(e)],2*patch['rowmap'][int(e)]+1]],int)
    P['Glo']=patch['Gminus'][np.ix_(rr,rr)]
    P['Ghi']=patch['Gplus'][np.ix_(rr,rr)]
    P['Egap']=P['Ghi']-P['Glo']
    P['Efac']=patch['Rfactor'][rr,:]
    P['Gt']=patch['Gexact'][np.ix_(rr,rr)]
textract=perf_counter()-t0

# RB: ordinary Galerkin snapshot QR from 27 out-of-sample pairs of (position, contrast).
# Includes (coarse) residual based compliance interval with a nominal lower bound
# for lambda_min(A0); do not mistake this for state of the art SCM RB.
t0=perf_counter();train=[]
for i in range(27):
    tid=list(Tiles)[i%len(Tiles)]; lo,hi=Tiles[tid]
    mu=lo+rng.uniform(0.12,0.88,2)*(hi-lo)
    Rm=np.where(np.linalg.norm(cent-mu,axis=1)<=RINC)[0]
    F=grad_rows(Rm)
    de=np.exp(rng.normal(0,.8,len(Rm)))*3 if i%2==0 else rng.uniform(-.8,-.1,len(Rm))*a0[Rm]
    Am=A0+F.T@sp.diags(np.repeat(de,2))@F
    train.append(sla.spsolve(Am.tocsc(),f))
Q,_=np.linalg.qr(np.column_stack(train),mode='reduced')
V=Q[:,:16]
K0=V.T@(A0@V);fv=V.T@f
# Obtain estimate for lambda_min(A0), deliberately factor 0.5; nominal numeric cert only.
lam_num=float(sla.eigsh(A0,k=1,which='SA',return_eigenvectors=False,tol=1e-9)[0]);lam0=.5*lam_num
for pos in positions:pos['FV']=pos['F']@V
trb=perf_counter()-t0

# global query preparation excluded from evaluation timings, but measured separately above.
results={}; times={}
def bench(name,func):
    vals=[];t0=perf_counter()
    for q in queries:vals.append(func(q,positions[q['pos_id']]))
    times[name]=perf_counter()-t0
    results[name]=vals

bench('direct',lambda q,p: float(f@sla.spsolve((A0+p['F'].T@sp.diags(q['da'])@p['F']).tocsc(),f)))
# exact tile Green. Corresponds to exact WB but no global solves online.
def wb(q,p,G):
    H=G+np.diag(1/q['da'])
    return float(c0-p['g']@np.linalg.solve(H,p['g']))
bench('exact_tile',lambda q,p: wb(q,p,p['Gt']))
# For reference separately exact Woodbury recomputed for EVERY query; no cache.
bench('woodbury_uncached',lambda q,p: wb(q,p,p['F']@lu0.solve(p['F'].T.toarray())))
# Cached exact G for each reused location: build on demand first call, not stored offline.
gcache={}
def cached(q,p):
    if q['pos_id'] not in gcache:
        mat=p['F']@lu0.solve(p['F'].T.toarray())
        gcache[q['pos_id']]=(mat+mat.T)/2
    return wb(q,p,gcache[q['pos_id']])
bench('woodbury_cached',cached)

certwidth=[];certok=[];errors=[]; certfails=0
def lower(q,p):
    global certfails
    Dinv=np.diag(1/q['da']);Nm=Dinv+p['Glo'];Np=Dinv+p['Ghi']
    if np.count_nonzero(np.linalg.eigvalsh(Nm)<0)!=np.count_nonzero(np.linalg.eigvalsh(Np)<0):
        certfails+=1;return (np.nan,np.nan)
    return float(c0-p['g']@np.linalg.solve(Nm,p['g'])),float(c0-p['g']@np.linalg.solve(Np,p['g']))
bench('loewner',lower)
# One-solve certified enclosure for positive coefficients: D in [Dminus-w^T E w,Dminus]
# For nonpositive/mixed contrast, fall back to the two-solve/inertia-check enclosure.
def hybrid(q,p):
    Dinv=np.diag(1/q['da']);Nm=Dinv+p['Glo']
    if np.all(q['da']>0):
        w=np.linalg.solve(Nm,p['g']); Dup=float(p['g']@w)
        gap=max(0.,float(w@(p['Egap']@w)))
        return float(c0-Dup),float(c0-Dup+gap)
    Np=Dinv+p['Ghi']
    if np.count_nonzero(np.linalg.eigvalsh(Nm)<0)!=np.count_nonzero(np.linalg.eigvalsh(Np)<0):
        return (np.nan,np.nan)
    return float(c0-p['g']@np.linalg.solve(Nm,p['g'])),float(c0-p['g']@np.linalg.solve(Np,p['g']))
bench('loewner_hybrid',hybrid)
# Factorized gap E = R R^T. One-solve positive: gap=||R^T w||^2.
def hybrid_factored(q,p):
    Dinv=np.diag(1/q['da']); Nm=Dinv+p['Glo']
    if np.all(q['da']>0):
        w=np.linalg.solve(Nm,p['g']);Dup=float(p['g']@w)
        gap=float(np.linalg.norm(p['Efac'].T@w)**2)
        return float(c0-Dup),float(c0-Dup+gap)
    Np=Nm+p['Efac']@p['Efac'].T
    if np.count_nonzero(np.linalg.eigvalsh(Nm)<0)!=np.count_nonzero(np.linalg.eigvalsh(Np)<0):
        return (np.nan,np.nan)
    return float(c0-p['g']@np.linalg.solve(Nm,p['g'])),float(c0-p['g']@np.linalg.solve(Np,p['g']))
bench('loewner_hybrid_factored',hybrid_factored)
# rb lower J and coarse upper J (pointwise alpha).
def rb(q,p):
    FV=p['FV']; K=K0+FV.T@(q['da'][:,None]*FV)
    y=np.linalg.solve(K,fv);u=V@y
    Jr=float(fv@y)
    res=f-A0@u-p['F'].T@(q['da']*(FV@y))
    a_new=a0[p['Rm']]+q['da'][::2];alpha=min(1.,float(np.min(a_new/a0[p['Rm']])))
    width=float(res@res)/(alpha*lam0)
    return Jr,Jr+width
bench('rb16',rb)

# Numeric quality: compare to exact tile Woodbury (algebraic reference, not validated floating point).
ref=np.array(results['exact_tile'])
for i, (l,u) in enumerate(results['loewner']):
    if not np.isfinite(l):continue
    certwidth.append((u-l)/max(abs(ref[i]),1e-14))
    certok.append(l-2e-9*max(abs(ref[i]),1.)<=ref[i]<=u+2e-9*max(abs(ref[i]),1.))
factvals=np.array(results['loewner_hybrid_factored'])
factwidth=(factvals[:,1]-factvals[:,0])/np.maximum(np.abs(ref),1e-14)
factvalid=np.mean((factvals[:,0]-2e-9*np.maximum(abs(ref),1)<=ref)&(ref<=factvals[:,1]+2e-9*np.maximum(abs(ref),1)))
hybridvals=np.array(results['loewner_hybrid'])
hybridwidth=(hybridvals[:,1]-hybridvals[:,0])/np.maximum(np.abs(ref),1e-14)
hybridvalid=np.mean((hybridvals[:,0]-2e-9*np.maximum(abs(ref),1)<=ref)&(ref<=hybridvals[:,1]+2e-9*np.maximum(abs(ref),1)))
rbvals=np.array(results['rb16']); rb_width=np.maximum(rbvals[:,1]-rbvals[:,0],0)/(np.maximum(abs(ref),1e-14))
rb_ok=np.mean((rbvals[:,0]-1e-9<=ref)&(ref<=rbvals[:,1]+1e-9))
numdiff={m:float(np.max(np.abs(np.asarray(results[m])-ref)/np.maximum(np.abs(ref),1e-14))) for m in ['direct','woodbury_uncached','woodbury_cached']}

# Timed elapsed cumulative prefix for variants on consistent query ordering, report average
# based on aggregate full runs; offline break-even against direct, cached and exact tile.
Q=len(queries)
to_lo=tgeom+trank+textract
to_ex=tgeom+texact+textract
means={name:times[name]/Q for name in times}

def break_even(a,b):
    ta={'loewner':to_lo,'exact_tile':to_ex,'rb16':trb}.get(a,0)
    tb={'loewner':to_lo,'exact_tile':to_ex,'rb16':trb}.get(b,0)
    den=means[b]-means[a]
    return (ta-tb)/den if den>0 else None
out=dict(problem=dict(n_free=Nf,tiles=len(Tiles),positions=len(positions),queries=Q, mesh_m=MESH_M,inclusion_radius=RINC,collar=ELL,k=K,rb_rank=V.shape[1],p=[int(2*len(p['Rm'])) for p in positions]),
        offline_s=dict(common_A0=tcommon,local_geometry=tgeom,loewner_enrichment=trank,exact_tile_enrichment=texact,row_extraction=textract,rb16_training=trb, loewner_total_ex_common=to_lo,exact_tile_total_ex_common=to_ex),
        global_rhs=dict(loewner=count_lo,exact_tile=count_ex),
        online_s_total=times,online_ms_per_query={n:1000*v/Q for n,v in times.items()},
        factorized_checks=dict(valid_fraction=float(factvalid),relwidth_median=float(np.nanmedian(factwidth)),relwidth_max=float(np.nanmax(factwidth))),
        factorized_storage_MB=float(sum(P['Gminus'].nbytes+P['Rfactor'].nbytes for P in patches.values())/1e6),
        dense_bracket_storage_MB=float(sum(P['Gminus'].nbytes+P['Gplus'].nbytes for P in patches.values())/1e6),
        exact_tile_storage_MB=float(sum(P['Gexact'].nbytes for P in patches.values())/1e6),
        hybrid_checks=dict(valid_fraction=float(hybridvalid),relwidth_median=float(np.nanmedian(hybridwidth)),relwidth_max=float(np.nanmax(hybridwidth))),
        loewner_checks=dict(certified=int(Q-certfails),total=Q,valid_vs_exact_tile=int(sum(certok)),relwidth_median=float(np.median(certwidth)) if certwidth else None,relwidth_max=float(np.max(certwidth)) if certwidth else None),
        rb16_check=dict(valid_fraction=float(rb_ok),relwidth_median=float(np.median(rb_width)),relwidth_max=float(np.max(rb_width)),nominal_lambda0=lam0),
        max_relative_difference_to_exact_tile=numdiff,
        break_even_queries=dict(lo_vs_direct=break_even('loewner','direct'),lo_vs_cached_wb=break_even('loewner','woodbury_cached'),lo_vs_exact_tile=break_even('loewner','exact_tile'),exact_tile_vs_direct=break_even('exact_tile','direct')))
print(json.dumps(out,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
open(f'monotonicity_manyquery_fair_results_m{MESH_M}.json','w').write(json.dumps(out,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
