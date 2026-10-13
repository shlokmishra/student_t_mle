"""Small before/after selector audit; no sampler or posterior simulations.

Uses the preserved original source and independent validation oracle. Inputs,
seeds, warmed timing repetitions and ceilings are in the JSON configuration.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from models.loc.student import mle as corrected
from scripts.selected_mle_reference_oracle import selected_mle_oracle


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run_estimate(module,x,k,tol):
    start=time.perf_counter()
    try:
        if module is corrected:
            value,diagnostics=module.global_mle(x,k,location_tol=tol,return_diagnostics=True)
        else:
            value=module.global_mle(x,k,location_tol=tol);diagnostics={}
        if not np.isfinite(value):return {'error':'nonfinite result','seconds':time.perf_counter()-start}
        return {'selected':float(value),'score_residual':abs(corrected.score(x,value,k)),
                'loglik':corrected.loglik(x,value,k),'diagnostics':diagnostics,'seconds':time.perf_counter()-start}
    except Exception as exc:
        return {'error':str(exc),'type':type(exc).__name__,'seconds':time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output'];out.mkdir(exist_ok=True,parents=True)
    baseline_path=ROOT/cfg['baseline']
    spec=importlib.util.spec_from_file_location('student_mle_before_hardening',baseline_path)
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    warm=[]
    for module in (original,corrected):
        start=time.perf_counter();module.global_mle([-1.,.25,.75],2.)
        warm.append({'module':module.__name__,'first_call_seconds':time.perf_counter()-start})
    timings=[];oracle_rows=[];start_phase=time.monotonic()
    for group in cfg['benchmark_groups']:
        rng=np.random.default_rng(group['seed']);data=rng.standard_t(group['k'],size=(group['draws'],group['n']))
        estimates={label:[run_estimate(module,x,group['k'],cfg['location_tol']) for x in data]
                   for label,module in [('original',original),('corrected',corrected)]}
        seconds={'original':[],'corrected':[]}
        for repetition in range(cfg['timing_repetitions']):
            # Alternate order so the corrected version is not always timed last.
            methods=[('original',original),('corrected',corrected)]
            if repetition%2:methods.reverse()
            for label,module in methods:
                tic=time.perf_counter()
                for x in data:
                    try:module.global_mle(x,group['k'],location_tol=cfg['location_tol'])
                    except corrected.StudentMLEError:pass
                seconds[label].append(time.perf_counter()-tic)
        timings.append({'group':group,'batch_seconds':seconds,
                        'median_seconds':{label:float(np.median(t)) for label,t in seconds.items()},
                        'fallback_count':sum(r.get('diagnostics',{}).get('fallback',False) for r in estimates['corrected']),
                        'failure_count':{label:sum('error' in r for r in rows) for label,rows in estimates.items()},
                        'failures':[{'index':i,'data':data[i].tolist(),'record':r} for i,r in enumerate(estimates['corrected']) if 'error' in r]})
        if group['n']<=10:
            for index,x in enumerate(data[:cfg['oracle_checks_per_group']]):
                oracle=selected_mle_oracle(x,group['k'],fast=True)
                row={'group':group,'index':index,'data':x.tolist(),'oracle':oracle}
                for label in estimates:
                    r=dict(estimates[label][index])
                    if 'error' not in r:
                        r.update(location_error=abs(r['selected']-oracle['selected']),
                                 loglik_gap=max(0.,oracle['loglik']-r['loglik']))
                    row[label]=r
                oracle_rows.append(row)
        save(out/'benchmark.json',{'config':cfg,'warmup':warm,'timings':timings})
        save(out/'ordinary_oracle_comparison.json',oracle_rows)
        print('benchmark',group, timings[-1],flush=True)
        if time.monotonic()-start_phase>cfg['ceiling_seconds']:raise TimeoutError('benchmark ceiling')
    prior=ROOT/'docs/research/evidence/selected_mle_reference_2026-10-03'
    cases=json.loads((prior/'selector_audit.json').read_text())
    cases+=json.loads((prior/'selector_near_tie_diagnosis.json').read_text())
    cases.append({'case':'centering_information_loss','data':[0.,1e-4,1e14,2e14,3e14],'k':1.})
    audit=[]
    for index,case in enumerate(cases):
        x=np.array(case['data']);k=case['k'];oracle=selected_mle_oracle(x,k,fast=True)
        row={'case':case.get('case',str(index)),'data':x.tolist(),'k':k,'oracle':oracle}
        for label,module in [('original',original),('corrected',corrected)]:
            result=run_estimate(module,x,k,cfg['location_tol'])
            if 'error' not in result:
                result.update(location_error=abs(result['selected']-oracle['selected']),loglik_gap=max(0.,oracle['loglik']-result['loglik']))
            try:
                roots=module.stationary_points(x,k,location_tol=cfg['location_tol'])
                result['roots']=roots.tolist()
                result['missing_roots']=int(sum(np.min(np.abs(roots-r))>max(cfg['location_tol'],1e-12) for r in oracle['roots']))
                result['root_count_error']=len(roots)-len(oracle['roots'])
            except Exception as exc:result['root_error']=str(exc)
            row[label]=result
        audit.append(row);save(out/'adversarial_comparison.json',audit)
        if time.monotonic()-start_phase>cfg['ceiling_seconds']:raise TimeoutError('audit ceiling')
    extreme=[]
    for x in ([0.,1.,1e200],[-1e200,1e200], [0.,2.**40], [2.**40,2.**40+.25,2.**40+1.5]):
        extreme.append({'data':x,'k':2.,'original':run_estimate(original,np.array(x),2.,cfg['location_tol']),
                        'corrected':run_estimate(corrected,np.array(x),2.,cfg['location_tol'])})
    save(out/'extreme_comparison.json',extreme)
    save(out/'completion.json',{'elapsed_seconds_after_warmup':time.monotonic()-start_phase,'ordinary_oracle_checks':len(oracle_rows),'adversarial_checks':len(audit),'complete':True})
    print('audit complete',len(audit),'ordinary oracle',len(oracle_rows),flush=True)


if __name__=='__main__':main()
