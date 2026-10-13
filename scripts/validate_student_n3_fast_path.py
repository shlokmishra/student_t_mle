"""One bounded matched-path study; reuse validated raw n=3 reference banks."""
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from models.loc.student import selected_block as guarded
from models.loc.student.n3_endpoint import certify_endpoint
from scripts.followup_shifted_student_reference import load,save,summary,diagnostics,oracle_check,weight_particles
from scripts.validate_student_selected_mle_gibbs import sha
from scripts.study_student_n3_selected_geometry import categorical,CENTRAL,case

OUT=ROOT/'docs/research/evidence/student_n3_fast_path_2026-10-10'
RAW=ROOT/'tmp/student_n3_fast_path_20261010'


def references():
    previous=ROOT/'docs/research/evidence/student_n3_geometry_selected_mle_2026-10-10'
    old=load(previous/'new_block_comparisons.json')
    refs={r['k']:r['reference'] for r in old}
    refs[2]=load(ROOT/'docs/research/evidence/student_selected_mle_gibbs_2026-10-10/centered_n3_reference.json')
    fp=load(ROOT/'docs/research/evidence/student_selected_mle_gibbs_2026-10-10/fresh_n3_provenance.json')
    refs[2].update(raw_path=fp['raw_path'],sha256=fp['sha256'])
    saved=load(previous/'saved_sampler_geometry.json')
    # Recompute chamber probabilities from the actual raw weighted particles below.
    result={}
    for k,r in refs.items():
        path=ROOT/r['raw_path']
        if sha(path)!=r['sha256']:raise ArithmeticError('reference hash mismatch')
        bank=np.load(path);mu,x,w,_=weight_particles(bank['base'],bank['mle'],case(k))
        actual=summary(mu,w,x,case(k))
        for key in actual:np.testing.assert_allclose(actual[key],r['summary'][key],atol=1e-12,rtol=1e-12)
        result[k]=(mu,x,w,{**r,**categorical(x,k,w),'summary':actual})
    return result


def run():
    cfg=load(OUT/'config.json');RAW.mkdir(exist_ok=True,parents=True)
    # Genuine pre-change source, rather than fresh timing against historical timings.
    spec=importlib.util.spec_from_file_location('models._n3_frozen_block',OUT/'original_block.py')
    original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
    refs=references();results=[];oracle=[];start=time.monotonic()
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'models/loc/student/selected_block.py',ROOT/'models/loc/student/n3_endpoint.py',ROOT/'models/loc/student/mle.py',OUT/'original_block.py',ROOT/'scripts/selected_mle_reference_oracle.py']}
    for k in cfg['k_values']:
        mu,x,w,ref=refs[k]
        starts=[]
        for i in range(4):
            if k==2:
                p=load(ROOT/f'docs/research/evidence/student_selected_mle_gibbs_2026-10-10/centered_n3_chain{i}_provenance.json')
            else:
                p=load(ROOT/f'docs/research/evidence/student_n3_geometry_selected_mle_2026-10-10/block_k{k}_chain{i}_provenance.json')
            starts.append((np.array(p['initial_x']),float(p['initial_mu'])))
        for length in cfg['block_lengths']:
            traces=[];old_seconds=new_seconds=0.;total_old={};total_new={};timers={};reasons={};all_equal=True
            for i,(xx,mm) in enumerate(starts):
                params=dict(n=3,k=float(k),initial_x=xx,initial_mu=mm,prior_mean=0.,prior_std=2.,
                    block_length=length,num_iterations_T=cfg['iterations'],proposal_std_mu=.45,proposal_std_z=.1,
                    wall_ceiling_seconds=cfg['chain_ceiling_seconds'],proposal_record_stride=1000,proposal_record_limit=4)
                seed=cfg['seed']+100*k+10*length+i
                # Warm both complete paths, excluding compilation and cold starts.
                for module,fast in [(original,False),(guarded,True)]:
                    module.run_gibbs(seed,0.,{**params,'num_iterations_T':32,'n3_endpoint_fast_path':fast},False)
                pair={}
                for mode in (['original','guarded'] if i%2==0 else ['guarded','original']):
                    module=original if mode=='original' else guarded
                    stem=f'k{k}_m{length}_chain{i}_{mode}';path=RAW/(stem+'.npz')
                    if path.exists():raise FileExistsError(path)
                    try:r=module.run_gibbs(seed,0.,{**params,'n3_endpoint_fast_path':mode=='guarded'},False)
                    except Exception as exc:
                        if hasattr(exc,'partial'):np.savez_compressed(RAW/(stem+'_partial.npz'),**exc.partial)
                        save(OUT/(stem+'_failure.json'),dict(error=repr(exc),iteration=getattr(exc,'block_iteration',None)));raise
                    pair[mode]=r
                    np.savez_compressed(path,mu=r['mu_chain'],x=r['x_chain'])
                    provenance={key:v for key,v in r.items() if key not in ('mu_chain','x_chain')}
                    provenance.update(initial_x=xx.tolist(),initial_mu=mm,raw_path=str(path.relative_to(ROOT)),sha256=sha(path),source_sha256=sources)
                    save(OUT/(stem+'_provenance.json'),provenance)
                a,b=pair['original'],pair['guarded']
                equal=np.array_equal(a['mu_chain'],b['mu_chain']) and np.array_equal(a['x_chain'],b['x_chain'])
                if not equal:raise ArithmeticError('matched chain changed')
                all_equal &= equal;old_seconds+=a['seconds'];new_seconds+=b['seconds']
                for counts,total in [(a['counts'],total_old),(b['counts'],total_new)]:
                    for name,v in counts.items():total[name]=total.get(name,0)+v
                for name,v in b['times'].items():timers[name]=timers.get(name,0)+v
                for name,v in b['n3_fast_reasons'].items():reasons[name]=reasons.get(name,0)+v
                burn=cfg['burn_in'];traces.append((b['mu_chain'][burn+1:],b['x_chain'][burn+1:]))
                if i==0:
                    for index in [burn+1,cfg['iterations']]:
                        data=b['x_chain'][index];check=oracle_check(data,case(k),f'guarded_k{k}_m{length}',index,normalized=True)
                        g=guarded.geometry(data,0.,k)
                        check['certificate']=asdict(certify_endpoint(data,0.,k,1e-10,g.curvature,g.score));oracle.append(check)
                    proposal=b['proposal_records'][0]
                    oracle.append(oracle_check(proposal['proposed_x'],case(k),f'proposal_k{k}_m{length}',proposal['iteration']))
            mm=np.array([v[0] for v in traces]);xx=np.array([v[1] for v in traces]);maximum=np.max(abs(xx),axis=2)
            pooled=summary(mm.ravel(),np.ones(mm.size),xx.reshape(-1,3),case(k));batch=[];occbatch=[];chain=[];chainocc=[]
            for a,b in traces:
                chain.append(summary(a,np.ones(len(a)),b,case(k)));chainocc.append(categorical(b,k)['probabilities'])
                for lo in range(0,len(a),cfg['batch_length']):
                    sl=slice(lo,lo+cfg['batch_length']);batch.append(summary(a[sl],np.ones(len(a[sl])),b[sl],case(k)));occbatch.append(categorical(b[sl],k)['probabilities'])
            se={key:max(float(np.std([r[key] for r in batch],ddof=1)/np.sqrt(len(batch))),float(np.std([r[key] for r in chain],ddof=1)/2)) for key in pooled}
            occ=categorical(xx.reshape(-1,3),k)
            occse={key:max(float(np.std([r[key] for r in occbatch],ddof=1)/np.sqrt(len(occbatch))),float(np.std([r[key] for r in chainocc],ddof=1)/2)) for key in CENTRAL}
            z={key:(pooled[key]-ref['summary'][key])/math.hypot(se[key],ref['uncertainty'][key]['se']) if ref['uncertainty'][key]['se'] is not None else None for key in pooled}
            occz={key:(occ['probabilities'][key]-ref['probabilities'][key])/math.hypot(occse[key],ref['snis_mcse'][key]) for key in CENTRAL}
            ds=dict(mu=diagnostics(mm),maximum=diagnostics(maximum),tail6=diagnostics((maximum>6).astype(float)))
            row=dict(k=k,block_length=length,identical_chains=bool(all_equal),original_seconds=old_seconds,guarded_seconds=new_seconds,
                speedup=old_seconds/new_seconds,original_counts=total_old,guarded_counts=total_new,guarded_timers=timers,fast_reasons=reasons,
                reference=ref,summary=pooled,mcse=se,summary_difference_mcse=z,diagnostics=ds,occupancy=occ,chamber_mcse=occse,chamber_difference_mcse=occz,
                per_chain_occupancy=chainocc,per_chain_summary=chain,
                original_ess_per_second={key:d['ess_bulk']/old_seconds for key,d in ds.items()},guarded_ess_per_second={key:d['ess_bulk']/new_seconds for key,d in ds.items()},
                screen_summary=max(abs(v) for v in z.values() if v is not None)<=4,
                screen_chambers=max(abs(v) for v in occz.values())<=4,
                screen_diagnostics=all(d['rhat'] is not None and d['rhat']<=1.01 and d['ess_bulk'] is not None and d['ess_bulk']>=400 and d['ess_tail'] is not None and d['ess_tail']>=400 for d in ds.values()))
            results.append(row);save(OUT/'comparisons.json',results);save(OUT/'oracle_checks.json',oracle)
            print('completed',k,length,'speedup',row['speedup'],'fallbacks',total_new['n3_fast_fallbacks'],'screens',row['screen_summary'],row['screen_chambers'],row['screen_diagnostics'],flush=True)
            if time.monotonic()-start>cfg['total_ceiling_seconds']:raise TimeoutError('total study ceiling')
    if any('error' in r or r['selected_location_error']>1e-10 or r.get('released_mle_residual',0)>1e-10 for r in oracle):raise ArithmeticError('independent oracle failure')
    save(OUT/'execution.json',dict(seconds=time.monotonic()-start,source_sha256=sources,failures=0,chains=48,blocks=48*cfg['iterations'],oracle_checks=len(oracle),all_chains_identical=True))


if __name__=='__main__':run()
