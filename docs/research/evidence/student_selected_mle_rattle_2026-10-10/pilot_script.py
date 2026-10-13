import os,sys,json,time
from pathlib import Path
root=Path('/scratch/user/shlokmishra08/code/student_t_mle');sys.path.insert(0,str(root))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from models import loc_student_selected_mle_rattle as r
old=root/'docs/research/evidence/student_selected_mle_gibbs_2026-10-10'
out=root/'docs/research/evidence/student_selected_mle_rattle_2026-10-10'
cfg=json.loads((old/'config.json').read_text());results=[]
for case in cfg['cases']:
 p0=json.loads((old/(case['id']+'_chain1_provenance.json')).read_text())
 p={**case,'initial_x':p0['initial_x'],'initial_mu':p0['initial_mu'],'num_iterations_T':500,'wall_ceiling_seconds':90.,'mle_tolerance':1e-10,'rattle_step_size':.1,'rattle_num_steps':10}
 start=time.perf_counter();a=r.run_rattle(2026102100+case['n'],case['release'],p,False)
 a.pop('mu_chain');a.pop('x_chain');results.append({'case':case['id'],'elapsed_with_setup':time.perf_counter()-start,**a})
 (out/'pilot.json').write_text(json.dumps(results,indent=2,default=lambda x:x.tolist()))
 print(case['id'],a['seconds'],a['counts'],flush=True)
