"""Small post-run direct-bank oracle checks and source/array integrity checks."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scripts.followup_shifted_student_reference import load,save,oracle_check
from scripts.validate_student_selected_mle_gibbs import sha


def run():
    out=ROOT/'docs/research/evidence/student_independence_geometry_2026-10-10';cfg=load(out/'config.json');raw=ROOT/cfg['raw_directory']
    from diagnostics.source_provenance import verify_recorded_sources
    source_checks = verify_recorded_sources(ROOT, load(out/'preflight.json')['source_sha256'])
    source_checks += verify_recorded_sources(ROOT, load(out/'higher_n_execution.json')['source_sha256'])
    provenance=list(out.glob('*_chain*_provenance.json'))
    for p in provenance:
        d=load(p)
        if sha(ROOT/d['raw_path'])!=d['sha256']:raise ArithmeticError('raw chain changed')
    b=np.load(raw/'direct_n10.npz');base,mle=b['base'],b['mle'];indices=[0,int(np.argmax(np.max(abs(base),axis=1)))];checks=[]
    centered,shifted=cfg['cases']
    for index in indices:checks.append(oracle_check(base[index],centered,'fresh_direct_base',index,ceiling=15.))
    for index in [int(np.argmax(-.5*((shifted['release']-mle-shifted['prior_mean'])/shifted['prior_std'])**2)),indices[1]]:
        x=base[index]+shifted['release']-mle[index]
        checks.append(oracle_check(x,shifted,'fresh_direct_shifted_normalized',index,normalized=True,ceiling=15.))
    save(out/'direct_oracle_checks.json',checks)
    if any('error' in r or r['selected_location_error']>1e-10 or r.get('released_mle_residual',0)>1e-10 for r in checks):raise ArithmeticError('direct oracle check failed')
    save(out/'integrity_check.json',dict(sampling_sources_match=True,geometry_sources_match=True,source_checks=source_checks,raw_chains_verified=len(provenance),extra_direct_verification_selector_calls=len(checks)))
    print('verified',len(provenance),'raw chains and',len(checks),'additional direct oracle cases',flush=True)


if __name__=='__main__':run()
