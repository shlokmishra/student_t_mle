"""Exact n4/5 examples and finite-fold graph; no topology classification campaign."""
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import sympy as sp
from reporting.diagnostics import student_higher_n_geometry as geo
from scripts.followup_shifted_student_reference import save,oracle_check
from scripts.validate_student_selected_mle_gibbs import sha
from models.loc.student.mle import stationary_points


def run():
    out=ROOT/'docs/research/evidence/student_independence_geometry_2026-10-10';start=time.monotonic()
    save(out/'higher_n_graphs.json',[geo.graph_summary(n) for n in (3,4,5)])
    rows=geo.exact_examples();checks=[]
    for row in rows:
        x=np.array([float(sp.N(sp.sympify(v),40)) for v in row['exact_data']]);n=len(x)
        case=dict(model='student',n=n,k=1.,release=0.,prior_mean=0.,prior_std=2.)
        r=oracle_check(x,case,row['name'],n,ceiling=15.)
        if 'error' in r:save(out/'higher_n_failure.json',r);raise ArithmeticError(r['error'])
        roots=stationary_points(x,1.);r.update(exact=row,production_stationary_roots=list(map(float,roots)))
        np.testing.assert_allclose(roots,r['oracle']['roots'],atol=1e-10,rtol=0)
        checks.append(r);save(out/'higher_n_examples.json',checks)
        print(row['name'],'selected',r['production_selected'],'roots',list(map(float,roots)),flush=True)
        if time.monotonic()-start>90:raise TimeoutError('bounded geometry ceiling')
    # Concrete n4 fold allowed with a same-sign OTHER outer coordinate, impossible at n3.
    scores=np.array([.5,.1,-.3,-.3]);signs=(0,1,0,0)
    inverse=lambda a,outer:(1+np.sqrt(1-4*a*a))/(2*a) if outer else 2*a/(1+np.sqrt(1-4*a*a))
    witness=np.array([1.,inverse(.1,True),inverse(-.3,False),inverse(-.3,False)])
    save(out/'higher_n_fold_witness.json',dict(scores=scores.tolist(),data=witness.tolist(),score_residual=float(np.sum(witness/(1+witness*witness))),
        adjacent_chambers=[list(signs),[1,1,0,0]],proof='1/2+1/10-3/10-3/10=0; both inner/outer first-coordinate sheets meet at the finite fold'))
    save(out/'higher_n_execution.json',dict(seconds=time.monotonic()-start,oracle_cases=len(checks),failures=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'reporting/diagnostics/student_higher_n_geometry.py',ROOT/'models/loc/student/mle.py']}))


if __name__=='__main__':run()
