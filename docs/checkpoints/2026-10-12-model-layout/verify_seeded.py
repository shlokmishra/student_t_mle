import importlib,json,os,pathlib,sys
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
root=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(root));import numpy as np
checkpoint=pathlib.Path(__file__).resolve().parent
mapping={
'loc_student':'loc.student.score_gibbs','loc_student_rattle':'loc.student.score_rattle',
'loc_student_selected_mle':'loc.student.selected_gibbs','loc_student_selected_mle_block':'loc.student.selected_block',
'loc_student_selected_mle_rattle':'loc.student.selected_rattle','loc_student_selected_mle_independence':'loc.student.independence',
'loc_affine':'loc.affine','loc_logistic':'loc.logistic.score_gibbs','loc_laplace':'loc.laplace.median_gibbs',
'loc_scale_student':'loc_scale.student_gibbs','loc_scale_student_rattle':'loc_scale.student_rattle'}
phase='after';arrays={};summary={}
for name,new in mapping.items():
 m=importlib.import_module('models.'+(name if phase=='before' else new))
 p=dict(n=3,k=2.,num_iterations_T=12,proposal_std_mu=.2,proposal_std_z=.1,prior_mean=0.,prior_std=2.,initial_x=np.array([-.4,.4,0.]),initial_mu=.1,initial_e=np.array([-1.,.2,2.]),block_length=4,n3_endpoint_fast_path=True,gibbs_backend='numba_full',rattle_step_size=.005,rattle_num_steps=2,proposal_std_logsigma=.1,prior_logsigma_mean=0.,prior_logsigma_std=1.,slice_max_shrink_steps=200)
 if name=='loc_affine':p.update(model='student',tau_q=.3)
 if name.startswith('loc_scale'):p['n']=10
 if name.endswith('independence'):result=m.run_independence(np.array([0,20261012],dtype=np.uint32),0.,p,False)
 elif name.startswith('loc_scale'):result=(m.run_rattle if 'rattle' in name else m.run_gibbs)(np.array([0,20261012],dtype=np.uint32),0.,1.,p,False)
 else:result=(m.run_rattle if 'rattle' in name else m.run_gibbs)(np.array([0,20261012],dtype=np.uint32),0.,p,False)
 for key in ['mu_chain','x_chain','sigma_chain','x_final']:
  if key in result:arrays[name+'__'+key]=np.asarray(result[key])
 summary[name]={key:result[key] for key in ['counts','target','mu_acceptance_rate','pair_acceptance_rate','acceptance_rate','parameter_acceptance_rate'] if key in result}
 print(name,flush=True)
# Verification only: do not overwrite saved pre/post evidence.
summary=json.loads(json.dumps(summary,default=lambda x:x.item()))
if phase=='after':
 with np.load(checkpoint/'seeded_before.npz') as old:
  assert set(old.files)==set(arrays)
  for key,value in arrays.items():np.testing.assert_array_equal(value,old[key],err_msg=key)
 assert json.loads((checkpoint/'seeded_before-decisions.json').read_text())==summary
 print('All seeded arrays and acceptance decisions identical',flush=True)
