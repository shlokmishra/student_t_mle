"""Reuse exact archived adversaries; independently check six shifted witnesses."""
import os
from pathlib import Path
import sys
from dataclasses import asdict
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from models.loc.student.n3_endpoint import certify_endpoint
from models.loc.student import selected_block as block
from scripts.followup_shifted_student_reference import load,save,oracle_check
from scripts.study_student_n3_selected_geometry import case
from scripts.validate_student_selected_mle_gibbs import sha


def audit():
    out=ROOT/'docs/research/evidence/student_n3_fast_path_2026-10-10'
    source=ROOT/'docs/research/evidence/student_n3_geometry_selected_mle_2026-10-10/stress_oracle.json'
    rows=[]
    for previous in load(source):
        x=np.array(previous['data']);k=previous['k'];y=x;d=k+y*y
        c=float(np.sum(y/d));j=float(np.sum((k-y*y)/(d*d)))
        cert=certify_endpoint(x,0.,k,1e-10,j,c)
        row=dict(source=previous['source'],k=k,data=x.tolist(),certificate=asdict(cert),oracle=previous['oracle'])
        try:
            old=block.selected_state(x,0.,k,mle_tolerance=1e-10)
            row['original_status']='valid' if old is not None else 'resolved_invalid'
        except Exception as exc:row.update(original_status='explicit_failure',original_error=repr(exc))
        if cert.status==1 and abs(previous['oracle']['selected'])>1e-10:raise ArithmeticError('false acceptance')
        if cert.status==-1 and abs(previous['oracle']['selected'])<=1e-10:raise ArithmeticError('false rejection')
        rows.append(row)
    fresh=[]
    for k in (1,2,3):
        for name,normalized in [('central',[-13/16,11/27,1/7]),('inferior_local_max',[-13/16,27/11,7])]:
            t=1.25 if name=='central' else -.75
            x=t+np.sqrt(k)*np.array(normalized);y=x-t;d=k+y*y
            cert=certify_endpoint(x,t,k,1e-10,float(np.sum((k-y*y)/(d*d))),float(np.sum(y/d)))
            check=oracle_check(x,{**case(k),'release':t},f'shifted_{name}',k)
            check['release']=t;check['certificate']=asdict(cert);fresh.append(check)
            if 'error' in check:raise ArithmeticError(check['error'])
            if cert.status==1 and abs(check['oracle']['selected']-t)>1e-10:raise ArithmeticError('false shifted acceptance')
            assert cert.status==(1 if name=='central' else -1)
    save(out/'endpoint_audit.json',dict(source_path=str(source.relative_to(ROOT)),source_sha256=sha(source),archived_cases=rows,
        reasons=dict(Counter(r['certificate']['reason'] for r in rows)),statuses=dict(Counter(r['certificate']['status'] for r in rows)),
        fresh_shifted_oracle=fresh,false_acceptances=0,false_rejections=0))
    print('audited',len(rows),'saved exact oracle cases and',len(fresh),'fresh shifted cases',flush=True)


if __name__=='__main__':audit()
