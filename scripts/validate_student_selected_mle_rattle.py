"""Bounded selected-MLE endpoint-RATTLE audit; legacy algorithms remain untouched."""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from models.loc.student import score_rattle as legacy
from models.loc.student import selected_rattle as selected
from models.loc.student import selected_gibbs as gibbs
from scripts.validate_student_selected_mle_gibbs import sha,particles
from scripts.followup_shifted_student_reference import load,save,summary,diagnostics,distances,oracle_check,longest_run,run_affine_chain


def old_paths(cfg):
    old=ROOT/cfg['gibbs_evidence'];return old,ROOT/load(old/'config.json')['raw_directory']


def starts(case,cfg):
    old,_=old_paths(cfg)
    for i in range(cfg['chains']):
        p=load(old/(case['id']+f'_chain{i}_provenance.json'))
        yield i,np.array(p['initial_x']),p['initial_mu'],p['origin']


def preflight(cfg,out):
    old,raw=old_paths(cfg);ocfg=load(old/'config.json')
    execution=load(old/'chains_execution.json')
    from diagnostics.source_provenance import verify_recorded_sources
    source_checks = verify_recorded_sources(ROOT, execution['source_sha256'],
        ['models/loc/student/mle.py', 'models/loc/student/selected_gibbs.py'])
    fp=load(old/'fresh_n3_provenance.json')
    if sha(ROOT/fp['raw_path'])!=fp['sha256']:raise ArithmeticError('n3 bank changed')
    provenance=[]
    for case in cfg['cases']:
        _,_,(mu,x,w,_)=particles(case,ocfg,raw)
        ref=load(old/(case['id']+'_reference.json'));actual=summary(mu,w,x,case)
        for key in actual:np.testing.assert_allclose(actual[key],ref['summary'][key],atol=1e-12,rtol=1e-12)
        save(out/(case['id']+'_reference.json'),ref)
        provenance.append({'case':case['id'],'reference_path':str((old/(case['id']+'_reference.json')).relative_to(ROOT)),
          'sha256':sha(old/(case['id']+'_reference.json')),'source_checks':source_checks,
          'status':'raw hashes, recorded selector provenance and every summary checked; original bootstrap uncertainty carried forward unchanged'})
        for i,_,_,_ in starts(case,cfg):
            gp=load(old/(case['id']+f'_chain{i}_provenance.json'))
            if sha(ROOT/gp['raw_path'])!=gp['sha256']:raise ArithmeticError('Gibbs comparison chain changed')
    save(out/'reference_provenance.json',provenance)


def chains(cfg,raw,out):
    preflight(cfg,out);checks=[]
    for case in cfg['cases']:
        for i,x,mu,origin in starts(case,cfg):
            stem=case['id']+f'_chain{i}';path=raw/(stem+'.npz')
            if path.exists():raise FileExistsError(path)
            check=oracle_check(x,case,stem+'_initializer',0,normalized=True);checks.append(check);save(out/'initializer_oracle_checks.json',checks)
            if 'error' in check or check['released_mle_residual']>cfg['mle_tolerance']:raise ArithmeticError('initializer oracle failed')
            params={**case,**cfg['rattle'],'initial_x':x,'initial_mu':mu,'num_iterations_T':cfg['iterations'],
                'wall_ceiling_seconds':cfg['chain_ceiling_seconds'],'score_tolerance':cfg['score_tolerance'],'mle_tolerance':cfg['mle_tolerance'],
                'proposal_record_stride':1000,'proposal_record_limit':12}
            try:r=selected.run_rattle(case['chain_seed']+i,case['release'],params,True)
            except Exception as exc:
                import traceback
                save(out/(stem+'_failure.json'),{'error':repr(exc),'traceback':traceback.format_exc(),'params':{**params,'initial_x':x.tolist()}});raise
            mus,xs=r.pop('mu_chain'),r.pop('x_chain');np.savez_compressed(path,mu=mus,x=xs)
            save(out/(stem+'_provenance.json'),{**r,'initial_x':x.tolist(),'initial_mu':mu,'origin':origin,'failures':0,
                'raw_path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            print('complete',stem,r['seconds'],r['counts'],flush=True)


def chamber_labels(x,k):
    # Existing chamber graph describes score geometry; it is not used as an MLE selector.
    from reporting.diagnostics.verify_student_n3_components import chamber_graph,connected_components,chamber_name
    components=connected_components(chamber_graph());lookup={node:j for j,cc in enumerate(components) for node in cc}
    names=[];ids=[]
    for y in x:
        bits=tuple(int(v) for v in (np.abs(y)>np.sqrt(k)));signs=tuple(int(np.sign(v)) if bit else 0 for bit,v in zip(bits,y))
        node=(bits,signs);names.append(chamber_name(node));ids.append(lookup.get(node,-1))
    return np.array(names),np.array(ids)


def analyze(cfg,raw,out):
    preflight(cfg,out);old,oraw=old_paths(cfg);ocfg=load(old/'config.json')
    comparisons=[];rows=[];checks=[]
    for case in cfg['cases']:
        ref=load(out/(case['id']+'_reference.json'));_,_,(rmu,rx,rw,_)=particles(case,ocfg,oraw)
        traces=[np.load(raw/(case['id']+f'_chain{i}.npz')) for i in range(cfg['chains'])]
        provenance=[load(out/(case['id']+f'_chain{i}_provenance.json')) for i in range(cfg['chains'])]
        mu=np.array([a['mu'][cfg['burn_in']+1:] for a in traces]);x=np.array([a['x'][cfg['burn_in']+1:] for a in traces])
        maximum=np.max(np.abs(x-case['release']),axis=2);full=summary(mu.ravel(),np.ones(mu.size),x.reshape(-1,case['n']),case)
        per_chain=[];batches=[];splits=[];movement=[];component_records=[]
        for i in range(cfg['chains']):
            per_chain.append(summary(mu[i],np.ones(len(mu[i])),x[i],case));half=len(mu[i])//2
            splits.append([summary(mu[i,s],np.ones(half),x[i,s],case) for s in [slice(0,half),slice(-half,None)]])
            for lo in range(0,len(mu[i]),cfg['batch_length']):
                hi=lo+cfg['batch_length'];batches.append(summary(mu[i,lo:hi],np.ones(hi-lo),x[i,lo:hi],case))
            event=maximum[i]>6;branch=np.abs(x[i]-case['release'])>np.sqrt(case['k']);argmax=np.argmax(np.abs(x[i]-case['release']),axis=1)
            movement.append({'tail6_entries':int(np.sum(~event[:-1]&event[1:])),'tail6_exits':int(np.sum(event[:-1]&~event[1:])),
                'longest_tail6_run':longest_run(event),'max_residual':float(maximum[i].max()),'branch_vector_changes':int(np.sum(np.any(branch[1:]!=branch[:-1],axis=1))),
                'largest_residual_identity_changes':int(np.sum(argmax[1:]!=argmax[:-1]))})
            if case['n']==3:
                names,ids=chamber_labels(x[i]-case['release'],case['k'])
                component_records.append({'chain':i,'chamber_counts':dict(Counter(names)),'component_counts':dict(Counter(map(int,ids))),
                    'chamber_changes':int(np.sum(names[1:]!=names[:-1])),'component_changes':int(np.sum(ids[1:]!=ids[:-1]))})
            indices=np.unique(np.r_[np.linspace(0,len(mu[i])-1,cfg['retained_oracle_checks'],dtype=int),np.argmax(maximum[i])])
            for idx in indices:checks.append(oracle_check(x[i,idx],case,case['id']+f'_retained{i}',idx,normalized=True))
            for record in provenance[i]['proposal_records']:
                check=oracle_check(record['proposed_x'],case,case['id']+f'_proposed{i}',record['iteration'])
                check['endpoint_status']=record['selected_endpoint'];check['rejection']=record['rejection']
                if record['selected_endpoint'] is True:check['released_mle_residual']=abs(check.get('oracle',{}).get('selected',np.inf)-case['release'])
                if record['selected_endpoint'] is False and 'oracle' in check and abs(check['oracle']['selected']-case['release'])<=cfg['mle_tolerance']:
                    check['error']='oracle disagrees with endpoint support rejection'
                checks.append(check)
        if case['n']==3:
            names,ids=chamber_labels(rx-case['release'],case['k']);weights={name:float(rw[names==name].sum()/rw.sum()) for name in np.unique(names)}
            save(out/'n3_components.json',{'chain_geometry':component_records,'reference_chamber_probabilities':weights,
                'reference_component_weight':{int(j):float(rw[ids==j].sum()/rw.sum()) for j in np.unique(ids)},
                'caveat':'graph geometry does not prove global-MLE classification; all retained states were checked by the hardened selector'})
        se={key:max(float(np.std([r[key] for r in batches],ddof=1)/np.sqrt(len(batches))),float(np.std([r[key] for r in per_chain],ddof=1)/np.sqrt(cfg['chains']))) for key in full}
        if case['id']=='shifted_n10':se['upper_tail']=None
        for key in ['lower_tail','upper_tail','latent_tail3','latent_tail6']:
            if all(r[key]==0 for r in per_chain) or all(r[key]==1 for r in per_chain):se[key]=None
        diag={'mu':diagnostics(mu),'latent_max':diagnostics(maximum),'latent_tail6':diagnostics((maximum>6).astype(float))}
        seconds=sum(p['seconds'] for p in provenance)
        for d in diag.values():d.update(ess_bulk_per_second=d['ess_bulk']/seconds if d['ess_bulk'] is not None else None,ess_tail_per_second=d['ess_tail']/seconds if d['ess_tail'] is not None else None)
        z={}
        for key in full:
            rs,cs=ref['uncertainty'][key]['se'],se[key];z[key]=(full[key]-ref['summary'][key])/np.hypot(rs,cs) if rs is not None and cs is not None and np.hypot(rs,cs)>0 else None
            rows.append({'case':case['id'],'metric':key,'reference':ref['summary'][key],'reference_mcse':rs,'rattle':full[key],'rattle_mcse_diagnostic':cs,'difference_mcse':z[key]})
        screens={name:d['rhat'] is not None and d['rhat']<=cfg['rhat_max'] and all(d[k] is not None and d[k]>=cfg['ess_min'] for k in ['ess_bulk','ess_tail']) for name,d in diag.items()}
        screens['summary_comparison']=all(z[key] is not None and abs(z[key])<=cfg['difference_mcse_max'] for key in full if not(case['id']=='shifted_n10' and key=='upper_tail'))
        counts={key:sum(p['counts'][key] for p in provenance) for key in provenance[0]['counts']};rates={key:counts[key]/counts['proposals'] for key in ['accepted','forward_projection','reverse_phase','nonpositive_curvature','cheap_metropolis','global_selection']}
        gp=[load(old/(case['id']+f'_chain{i}_provenance.json')) for i in range(cfg['chains'])]
        gc=next(r for r in load(old/'comparisons.json') if r['case']['id']==case['id'])
        result={'case':case,'summary':full,'mcse_diagnostic':se,'per_chain':per_chain,'split_summaries':splits,'movement':movement,
            'diagnostics':diag,'screens':screens,'passes_declared_screens':all(screens.values()),'difference_mcse':z,'retained':int(mu.size),
            'parameter_distances':distances(mu.ravel(),rmu,rw),'latent_max_distances':distances(maximum.ravel(),np.max(np.abs(rx-case['release']),axis=1),rw),
            'counts':counts,'rates':rates,'seconds':seconds,'microseconds_per_iteration':seconds/(cfg['chains']*cfg['iterations'])*1e6,
            'microseconds_per_forward_or_reverse_trajectory':sum(p['phase_seconds'] for p in provenance)/(counts['forward_trajectories']+counts['reverse_trajectories'])*1e6,
            'numerical_maxima':{key:max(p['numerical_maxima'][key] for p in provenance) for key in provenance[0]['numerical_maxima']},
            'gibbs':{'summary':gc['summary'],'diagnostics':gc['diagnostics'],'seconds':sum(p['seconds'] for p in gp),
              'ess_bulk_per_second':{key:value['ess_bulk']/sum(p['seconds'] for p in gp) for key,value in gc['diagnostics'].items()}},
            'rare_upper_events':int(np.sum(mu>case['release']+case['tail_width']))}
        comparisons.append(result);save(out/'comparisons.json',comparisons);save(out/'oracle_checks.json',checks)
        print('analyze',case['id'],diag,screens,flush=True)
    with (out/'summary.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if any('error' in r or r.get('released_mle_residual',0)>cfg['mle_tolerance'] or r.get('selected_location_error',0)>cfg['mle_tolerance'] for r in checks):raise ArithmeticError('independent oracle validation failed; see saved rows')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for row,case in enumerate(cfg['cases']):
        for i in range(cfg['chains']):
            a=np.load(raw/(case['id']+f'_chain{i}.npz'));stride=max(1,len(a['mu'])//1000)
            axes[row,0].plot(np.arange(len(a['mu']))[::stride],a['mu'][::stride],lw=.6,label=str(i))
            axes[row,1].plot(np.arange(len(a['mu']))[::stride],np.max(np.abs(a['x'][::stride]-case['release']),axis=1),lw=.6)
        axes[row,0].set_ylabel(case['id']+' mu');axes[row,1].set_ylabel('max residual');axes[row,1].set_yscale('log')
        for ax in axes[row]:ax.axvline(cfg['burn_in'],color='gray',ls=':');ax.set_xlabel('iteration')
    axes[0,0].legend(title='chain',ncol=4);fig.savefig(out/'traces.png',dpi=150);plt.close(fig)


def benchmark(cfg,raw,out):
    rows=[]
    for case in cfg['cases']:
        _,x,mu,_=next(starts(case,cfg));p={**case,**cfg['rattle'],'initial_x':x,'initial_mu':mu,'mle_tolerance':cfg['mle_tolerance'],'initialization':'random','store_x_chain':True}
        def run(name,steps):
            pp={**p,'num_iterations_T':steps}
            if name=='rattle':return selected.run_rattle(992,case['release'],pp,False)
            if name=='legacy_rattle':return legacy.run_rattle(np.array([0,992],dtype=np.uint32),case['release'],pp,False)
            if name=='gibbs':return gibbs.run_gibbs(992,case['release'],pp,False)
            return run_affine_chain(x,mu,992,{**case,'mle_tolerance':1e-8},steps)[1]
        names=['legacy_rattle','rattle','affine','gibbs'];timings={n:[] for n in names}
        for n in names:run(n,5)
        for rep in range(3):
            for n in names if rep%2==0 else names[::-1]:
                start=time.perf_counter();run(n,cfg['benchmark_iterations']);timings[n].append(time.perf_counter()-start)
        rows.append({'case':case['id'],'iterations':cfg['benchmark_iterations'],'seconds':timings,
            'median_microseconds_per_iteration':{n:float(np.median(v)/cfg['benchmark_iterations']*1e6) for n,v in timings.items()},
            'note':'warmed single-core three repeats; same RATTLE step size/length; legacy has a different target and weaker reverse check; Gibbs sweeps have floor(n/2) pair proposals; throughput is not statistical efficiency'})
    save(out/'benchmark.json',rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--phase',choices=['chains','analyze','benchmark'],required=True);args=ap.parse_args();cfg=load(args.config)
    raw,out=ROOT/cfg['raw_directory'],ROOT/cfg['evidence_directory'];raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    paths=['models/loc/student/selected_rattle.py','models/loc/student/score_rattle.py','models/loc/student/selected_gibbs.py','models/loc/student/mle.py','models/loc/student/mu_mh.py','models/loc/affine.py','scripts/validate_student_selected_mle_rattle.py','scripts/validate_student_selected_mle_gibbs.py','scripts/followup_shifted_student_reference.py']
    save(out/(args.phase+'_execution.json'),{'config_sha256':sha(args.config),'source_sha256':{p:sha(ROOT/p) for p in paths},'python':sys.version,'cpu_affinity':sorted(os.sched_getaffinity(0)),'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    globals()[args.phase](cfg,raw,out)
if __name__=='__main__':main()
