"""Reproduce fixed precision/projection probes and reconcile historical tables."""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'


def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')


def precision():
    import mpmath as mp
    mp.mp.dps=90
    rows=json.loads((OUT/'stress_oracle.json').read_text());out=[]
    for r in rows:
        if not any(t in r['source'] for t in ['flat','tie']) or r['k']!=1:continue
        x=list(map(mp.mpf,r['data']));k=mp.mpf(r['k'])
        score=sum(v/(k+v*v) for v in x);j=sum((k-v*v)/(k+v*v)**2 for v in x)
        selected=mp.mpf(r['oracle']['selected'])
        root=mp.findroot(lambda t:sum((v-t)/(k+(v-t)**2) for v in x),
            (selected-mp.mpf('1e-10'),selected+mp.mpf('1e-10')),solver='secant',verify=True)
        values=[]
        for v in r['oracle']['roots']:
            q=mp.mpf(v);jj=sum((k-(z-q)**2)/(k+(z-q)**2)**2 for z in x)
            if jj>0:values.append((q,-sum(mp.log(1+(z-q)**2/k) for z in x)))
        values.sort(key=lambda v:v[1],reverse=True)
        gap=values[0][1]-values[1][1] if len(values)>1 else None
        out.append({'case':r['source'],'score_at_zero_exact_binary':mp.nstr(score,40),
            'curvature_at_zero':mp.nstr(j,40),'mpmath_root':mp.nstr(root,40),
            'oracle_location_error':mp.nstr(abs(root-selected),20),
            'best_vs_second_local_max_loglik_gap':mp.nstr(gap,25) if gap is not None else None,
            'precision_digits':90,'qualification':'High precision numerical evaluation/refinement for exact binary inputs; not a global root-enumeration certificate.'})
    save('high_precision_stress.json',out)


def projection():
    import numpy as np
    from models import loc_student_rattle as legacy
    from models import loc_student_selected_mle_rattle as r
    from reporting.diagnostics.student_n3_selected_geometry import chamber,component
    from scripts.followup_shifted_student_reference import oracle_check
    y=np.zeros(3);p=np.array([-4.,-2.,6.])
    opts=r.options({'rattle_step_size':1.,'rattle_num_steps':1})
    yp,pp,ok,d=r.phase_proposal(y,p,0.,1.,opts)
    h0=legacy._hamiltonian(y,p,0.,1.,0.);h1=legacy._hamiltonian(yp,pp,0.,1.,0.)
    class Fixed:
        def standard_normal(self,n):return p.copy()
        def random(self):return 0.
    x,state,diag=r.rattle_step(y,0.,0.,1.,Fixed(),opts)
    assert ok and component(chamber(yp))=='T3-'
    assert np.array_equal(x,y) and diag['rejection']=='nonpositive_curvature'
    save('projection_probe.json',{'initial_x':y.tolist(),'initial_p':p.tolist(),'mu':0.,'k':1.,
        'options':opts,'endpoint':yp.tolist(),'endpoint_p':pp.tolist(),'phase_admissible':ok,
        'endpoint_component':component(chamber(yp)),'diagnostics':d,
        'h_score_initial':h0,'h_score_final':h1,'score_target_acceptance_probability':math.exp(min(0.,h0-h1)),
        'endpoint_curvature':r.geometry(yp,0.,1.).curvature,'selected_step_rejection':diag['rejection'],
        'selected_retained_equals_initial':bool(np.array_equal(x,y)),
        'endpoint_oracle':oracle_check(yp,dict(k=1.,release=0.),'constructed_rattle_projection',0),
        'qualification':'Numerical counterexample to universal finite-RATTLE component preservation; one predetermined one-step phase proposal and its replay, not a mixing/tuning campaign. Selected sampler correctly rejects the endpoint.'})


def historical():
    import numpy as np
    from reporting.diagnostics.student_n3_selected_geometry import chamber,component
    path=ROOT/'docs/checkpoints/2026-09-20/evidence/n3_restarts.csv'
    rows=list(csv.DictReader(path.open()));components=Counter();groups=Counter();max_score=0.
    for row in rows:
        k=float(row['k']);x=np.array([float(row[f'source_x{i}']) for i in [1,2,3]])
        c=component(chamber(x,k));components[c]+=1
        groups['central' if c=='C0' else 'tail']+=1
        max_score=max(max_score,abs(float(np.sum(x/(k+x*x)))))
        assert (c=='C0')==bool(int(row['initial_mle_indicator']))
    save('historical_restart_reconciliation.json',{'source':str(path.relative_to(ROOT)),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'initial_component_counts':dict(components),
        'initial_groups':dict(groups),'maximum_source_score_residual':max_score,
        'reported_group_changes':sum(int(r['mle_exit_count'])+int(r['mle_return_count']) for r in rows),
        'acceptance_range':[min(float(r['x_acceptance_rate']) for r in rows),max(float(r['x_acceptance_rate']) for r in rows)],
        'qualification':'Summary preserves initial coordinates and branch codes, but not complete signed trajectories. No named-tail-component constancy is inferred from non-MLE occupancy alone. Release zero follows the documented experiment.'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['precision','projection','historical']);a=ap.parse_args()
    globals()[a.phase]()
