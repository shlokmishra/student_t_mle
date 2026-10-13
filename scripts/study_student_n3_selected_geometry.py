"""Bounded n=3 geometry investigation; production samplers remain unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import sympy as sp
from models.loc.student import selected_block as block
from models.loc.student.mle import global_mle
from reporting.diagnostics import student_n3_selected_geometry as geo
from reporting.diagnostics.verify_student_n3_components import chambers,chamber_graph,connected_components
from scripts.followup_shifted_student_reference import load,save,summary,diagnostics,oracle_check,weight_particles,particle_bootstrap,weighted_quantiles
from scripts.validate_student_selected_mle_gibbs import sha


def case(k):
    return dict(id=f'k{k}',model='student',n=3,k=float(k),release=0.,prior_mean=0.,prior_std=2.,tail_width=.75)


def labels(x,k):
    return [geo.node_name(geo.chamber(y,k)) for y in x]


NODES=sorted(geo.node_name(n) for n in chambers())
CENTRAL=sorted(geo.node_name(n) for n in chambers() if geo.component(n)=='C0')


def categorical(x,k,w=None):
    names=labels(x,k);w=np.ones(len(names)) if w is None else np.asarray(w)
    w=w/w.sum();probs={n:float(w[np.array(names)==n].sum()) for n in NODES}
    comps=Counter()
    for y,a in zip(x,w):comps[geo.component(geo.chamber(y,k))]+=float(a)
    # Explicit SNIS variance for each indicator; empty categories alone do not certify zero mass.
    se={n:float(np.sqrt(np.sum(w*w*((np.array(names)==n).astype(float)-probs[n])**2))) for n in NODES}
    return {'probabilities':probs,'snis_mcse':se,'component_probabilities':dict(comps),
        'variance_qualification':'SNIS/indicator formula assumes independent particles; correlated chains require the separate chamber_mcse field'}


def transition_analysis(x,k):
    ns=[geo.chamber(y,k) for y in x];cc=[geo.component(n) for n in ns]
    graph=chamber_graph();moves=Counter((geo.node_name(a),geo.node_name(b)) for a,b in zip(ns[:-1],ns[1:]) if a!=b)
    return {'retained':len(x),'chambers_visited':len(set(ns)), 'component_counts':dict(Counter(cc)),
        'component_changes':sum(a!=b for a,b in zip(cc[:-1],cc[1:])),
        'chamber_changes':sum(moves.values()),'nonadjacent_chamber_changes':sum(a!=b and b not in graph[a] for a,b in zip(ns[:-1],ns[1:])),
        'transition_counts':[{'from':a,'to':b,'count':c} for (a,b),c in sorted(moves.items())]}


def algebra(cfg,raw,out):
    result=geo.discriminant_certificate();result['witnesses']=[]
    graph=chamber_graph();parts=connected_components(graph)
    result['graph']={'nodes':[{'name':geo.node_name(n),'component':geo.component(n)} for n in chambers()],
        'edges':sorted({tuple(sorted([geo.node_name(n),geo.node_name(v)])) for n,vs in graph.items() for v in vs}),
        'components':[[geo.node_name(n) for n in sorted(p)] for p in parts]}
    for node in chambers():
        y=geo.rational_witness(node);d,q=geo.exact_quartic(y)
        result['witnesses'].append({'chamber':geo.node_name(node),'component':geo.component(node),'rational_y':list(map(str,y)),
            'exact_score':str(sum(v/(1+v*v) for v in y)),'Q':str(q.as_expr()),'Q_real_roots_sturm':int(q.count_roots(-sp.oo,sp.oo)),
            'e2':str(sum(y[i]*y[j] for i in range(3) for j in range(i+1,3)))})
    save(out/'algebra.json',result);print('algebra',len(result['witnesses']),'exact witnesses',flush=True)


def saved(cfg,raw,out):
    sources=[];checks=[];rows=[]
    for method,folder,rawdir,burn,lengths in [
        ('pair_gibbs','student_selected_mle_gibbs_2026-10-10','student_selected_mle_gibbs_20261010',2000,[None]),
        ('rattle','student_selected_mle_rattle_2026-10-10','student_selected_mle_rattle_20261010',2000,[None]),
        ('block','cheap_selected_mle_gibbs_2026-10-10','cheap_selected_mle_gibbs_20261010_v2',1000,[1,4,16])]:
        ev=ROOT/'docs/research/evidence'/folder
        for length in lengths:
            arrays=[];chain=[];proposed=[];total=Counter()
            for i in range(4):
                stem='centered_n3'+(f'_m{length}' if length else '')+f'_chain{i}'
                provenance=load(ev/(stem+'_provenance.json'));path=ROOT/provenance['raw_path']
                if sha(path)!=provenance['sha256']:raise ArithmeticError('saved raw hash mismatch')
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                x=np.load(path)['x'][burn+1:];arrays.append(x);chain.append(transition_analysis(x,2.))
                total.update(provenance.get('counts',{}))
                if method=='pair_gibbs':
                    total.update(provenance.get('rejections',{}))
                    for key in ['selector_calls','selector_fallbacks','pair_acceptance_count']:
                        if key in provenance:total[key]+=provenance[key]
                for p in provenance.get('proposal_records',[]):
                    if p.get('proposed_x') is not None:
                        node=geo.chamber(p['proposed_x'],2.)
                        proposed.append({'chamber':geo.node_name(node),'component':geo.component(node),
                            'accepted':p.get('accepted'), 'rejection':p.get('rejection'),'selector_called':p.get('selector_called')})
            blocks=[];chainprobs=[]
            for a in arrays:
                chainprobs.append(categorical(a,2.)['probabilities'])
                for lo in range(0,len(a),250):blocks.append(categorical(a[lo:lo+250],2.)['probabilities'])
            se={n:max(float(np.std([r[n] for r in blocks],ddof=1)/np.sqrt(len(blocks))),float(np.std([r[n] for r in chainprobs],ddof=1)/2)) for n in CENTRAL}
            rows.append({'method':method,'block_length':length,'occupancy':categorical(np.concatenate(arrays),2.),'chamber_mcse':se,'per_chain_occupancy':chainprobs,'chains':chain,'counts':dict(total),'recorded_proposals':proposed})
    old=ROOT/'docs/research/evidence/student_selected_mle_gibbs_2026-10-10'
    provenance=load(old/'fresh_n3_provenance.json');path=ROOT/provenance['raw_path']
    if sha(path)!=provenance['sha256']:raise ArithmeticError('saved reference hash mismatch')
    a=np.load(path);mu,x,w,_=weight_particles(a['base'],a['mle'],case(2))
    reference={'source':str(path.relative_to(ROOT)),'sha256':sha(path),'draws':len(mu),'weight_ess':float(w.sum()**2/(w@w)), 'summary':summary(mu,w,x,case(2)),**categorical(x,2.,w)}
    for r in rows:
        r['chamber_difference_mcse']={n:(r['occupancy']['probabilities'][n]-reference['probabilities'][n])/np.hypot(r['chamber_mcse'][n],reference['snis_mcse'][n]) for n in CENTRAL}
    for idx in np.unique(np.r_[0,len(mu)//2,np.argmax(np.max(np.abs(x),axis=1))]):checks.append(oracle_check(x[idx],case(2),'saved_reference',idx,normalized=True))
    path=ROOT/'tmp/cheap_selected_mle_gibbs_20261010_v2/centered_n3_direct_rejection.npz'
    manifest=load(ROOT/'docs/research/evidence/cheap_selected_mle_gibbs_2026-10-10/artifact_manifest.json')
    expected=next(r['sha256'] for r in manifest['ignored_raw_arrays'] if r['path']==str(path.relative_to(ROOT)))
    if sha(path)!=expected:raise ArithmeticError('saved rejection array hash mismatch')
    x=np.load(path)['x'];reference['independent_rejection']={'draws':len(x),'raw_path':str(path.relative_to(ROOT)),'sha256':expected,**categorical(x,2.)}
    path=ROOT/'tmp/gibbs-evolution-mle-chain.csv'
    data=list(csv.DictReader(path.open()));x=np.array([[float(r[f'x_{i}']) for i in range(3)] for r in data])
    legacy={'method':'legacy_score_gibbs','k':1,'raw_path':str(path.relative_to(ROOT)),'sha256':sha(path),'warning':'Historical single displayed segment; different prior/run and no selected-target reference comparison.', 'occupancy':categorical(x,1.),'transitions':transition_analysis(x,1.)}
    mismatch=sum((geo.component(geo.chamber(y))=='C0')!=(r['mu_star_is_global_mle']=='True') for y,r in zip(x,data))
    legacy['recorded_global_indicator_disagreements']=mismatch
    save(out/'saved_sampler_geometry.json',{'selected_chains':rows,'selected_reference_k2':reference,'legacy_saved_segment':legacy,'sources':sources})
    save(out/'saved_oracle_checks.json',checks)
    print('saved',[(r['method'],r['block_length'],[c['chambers_visited'] for c in r['chains']]) for r in rows],flush=True)


def stress(cfg,raw,out):
    rows=[]
    for node in chambers():
        exact=geo.rational_witness(node)
        for k in cfg['k_values']:
            x=np.array([float(v*sp.sqrt(k)) for v in exact]);row=oracle_check(x,case(k),'rational_chamber_'+geo.node_name(node),k)
            row.update(normalized_exact=list(map(str,exact)),predicted_component=geo.component(node),k=k)
            rows.append(row);save(out/'stress_oracle.json',rows)
    def inverse(z,b):
        root=sp.sqrt(1-4*z*z)
        return (1+root)/(2*z) if b else 2*z/(1+root)
    specs=[('origin',(sp.S.Zero,)*3),('finite_fold',(sp.S.One,-2+sp.sqrt(3),-2+sp.sqrt(3)))]
    for scale in [12,20]:
        e=sp.Rational(1,10**scale);zs=(e,-sp.Rational(1,4),sp.Rational(1,4)-e)
        for bits in [(1,0,0),(1,1,1)]:specs.append((f'asymptote_{scale}_{bits}',tuple(inverse(z,b) for z,b in zip(zs,bits))))
    e=sp.Rational(1,10**12);zs=(sp.Rational(1,2)-e,-sp.Rational(1,5),-sp.Rational(3,10)+e)
    for bits in [(0,0,0),(1,1,1)]:specs.append(('near_fold_'+str(bits),tuple(inverse(z,b) for z,b in zip(zs,bits))))
    for power in [4,8]:
        h=sp.Rational(1,10**power);zs=(sp.Rational(1,2)-h*h/4,-sp.Rational(1,2)+h*h/8,h*h/8)
        specs.append((f'central_near_flat_{power}',tuple(inverse(z,b) for z,b in zip(zs,(0,1,1)))))
    for power in [4,6]:
        h=sp.Rational(1,10**power);a=1+h;b=16/(h*h);q=a/(1+a*a)+b/(1+b*b)
        specs.append((f'inferior_near_tie_{power}',(-inverse(q,0),a,b)))
    # A genuine degenerate stationary root exists only outside selected support.
    u=sp.Symbol('u');root=sp.CRootOf(u**3-15*u*u+7*u-1,0)
    a=sp.sqrt(root);c=sp.sqrt(2*root/(3*root-1));specs.append(('tail_degenerate',(-c,a,a)))
    for name,exact in specs:
        for k in cfg['k_values']:
            x=np.array([float(sp.N(v*sp.sqrt(k),40)) for v in exact]); row=oracle_check(x,case(k),name,k,ceiling=15.)
            row.update(normalized_exact=list(map(str,exact)),k=k,float_release_score=float(np.sum(x/(k+x*x))),
                numerical_component=geo.component(geo.chamber(x,k)),exact_factorization_not_assumed_for_float_data=True)
            try:
                state=block.selected_state(x,0.,k,mle_tolerance=1e-10,score_tolerance=1e-10)
                row['selected_endpoint_status']='valid' if state is not None else 'resolved_invalid'
            except Exception as exc:row['selected_endpoint_status']='explicit_numerical_failure';row['endpoint_error']=repr(exc)
            try:
                block.seed_score_rng(2026104500+k)
                y=x.copy();pair=block.score_pair_kernel(y,0,1,0.,0.,float(k),cfg['sigma_z'])
                row['pair_status']='completed';row['pair_inverse_error']=pair[3]
            except Exception as exc:row['pair_status']='explicit_numerical_failure';row['pair_error']=repr(exc)
            rows.append(row);save(out/'stress_oracle.json',rows)
    print('stress',len(rows),'oracle checks',sum('error' in r for r in rows),'oracle errors',flush=True)


def blocks(cfg,raw,out):
    results=[];checks=[]
    for k in cfg['new_chain_k_values']:
        spec=case(k);path=raw/f'reference_k{k}.npz'
        if path.exists():raise FileExistsError(path)
        rng=np.random.default_rng(cfg['reference_seed']+k);start=time.monotonic()
        base=rng.standard_t(k,(cfg['reference_draws'],3));mle=np.empty(len(base));fallbacks=0
        for i,e in enumerate(base):
            mle[i],d=global_mle(e,k,return_diagnostics=True);fallbacks+=int(d['fallback'])
            if time.monotonic()-start>cfg['reference_ceiling_seconds']:raise TimeoutError('reference ceiling')
        seconds=time.monotonic()-start;np.savez_compressed(path,base=base,mle=mle)
        mu,x,w,_=weight_particles(base,mle,spec);ref=summary(mu,w,x,spec);refse=particle_bootstrap(mu,w,x,spec,cfg['bootstrap_seed']+k,cfg['bootstrap_repetitions'])
        maxx=np.max(np.abs(x),axis=1);starts=[]
        for q in [.1,.5,.9]:
            value=weighted_quantiles(maxx,w,[q])[0];idx=int(np.argmin(abs(maxx-value)));starts.append((x[idx],mu[idx]))
        e=np.array([-12.,0.,0.]);starts.append((e-global_mle(e,k),0.))
        reference={'summary':ref,'uncertainty':refse,'draws':len(mu),'weight_ess':float(w.sum()**2/(w@w)),**categorical(x,k,w),
            'raw_path':str(path.relative_to(ROOT)),'sha256':sha(path),'seed':cfg['reference_seed']+k,'selector_fallbacks':fallbacks,'failures':0,'seconds':seconds}
        for idx in np.unique(np.r_[np.linspace(0,len(mu)-1,6,dtype=int),np.argmax(w),np.argmax(maxx)]):
            checks.append(oracle_check(x[idx],spec,f'fresh_ref_k{k}',idx,normalized=True))
        traces=[];prov=[]
        for i,(xx,mm) in enumerate(starts):
            path=raw/f'block_k{k}_chain{i}.npz'
            if path.exists():raise FileExistsError(path)
            params={**spec,'initial_x':xx,'initial_mu':float(mm),'num_iterations_T':cfg['iterations'],'block_length':cfg['block_length'],
                'proposal_std_mu':cfg['sigma_mu'],'proposal_std_z':cfg['sigma_z'],'wall_ceiling_seconds':cfg['chain_ceiling_seconds']}
            try:r=block.run_gibbs(cfg['chain_seed']+100*k+i,0.,params,False)
            except Exception as exc:
                if hasattr(exc,'partial'):np.savez_compressed(raw/f'block_k{k}_chain{i}_partial.npz',**exc.partial)
                save(out/f'block_k{k}_chain{i}_failure.json',{'error':repr(exc),'iteration':getattr(exc,'block_iteration',None)});raise
            mus,xs=r.pop('mu_chain'),r.pop('x_chain');np.savez_compressed(path,mu=mus,x=xs)
            r.update(raw_path=str(path.relative_to(ROOT)),sha256=sha(path),initial_x=xx.tolist(),initial_mu=float(mm));prov.append(r)
            save(out/f'block_k{k}_chain{i}_provenance.json',r)
            mus,xs=mus[cfg['burn_in']+1:],xs[cfg['burn_in']+1:];traces.append((mus,xs))
            for idx in [0,len(xs)//2,int(np.argmax(np.max(np.abs(xs),axis=1)))]:checks.append(oracle_check(xs[idx],spec,f'block_k{k}_chain{i}',idx,normalized=True))
        mm=np.array([r[0] for r in traces]);xx=np.array([r[1] for r in traces]);maximum=np.max(np.abs(xx),axis=2)
        pooled=summary(mm.ravel(),np.ones(mm.size),xx.reshape(-1,3),spec)
        batches=[];occbatches=[];chainocc=[];chainmeans=[]
        for a,b in traces:
            chainocc.append(categorical(b,k)['probabilities']);chainmeans.append(summary(a,np.ones(len(a)),b,spec))
            for lo in range(0,len(a),cfg['batch_length']):
                sl=slice(lo,lo+cfg['batch_length']);batches.append(summary(a[sl],np.ones(len(a[sl])),b[sl],spec));occbatches.append(categorical(b[sl],k)['probabilities'])
        se={key:max(float(np.std([r[key] for r in batches],ddof=1)/np.sqrt(len(batches))),float(np.std([r[key] for r in chainmeans],ddof=1)/2)) for key in pooled}
        occ=categorical(xx.reshape(-1,3),k);occse={key:max(float(np.std([r[key] for r in occbatches],ddof=1)/np.sqrt(len(occbatches))),float(np.std([r[key] for r in chainocc],ddof=1)/2)) for key in CENTRAL}
        z={key:(pooled[key]-ref[key])/np.hypot(se[key],refse[key]['se']) if refse[key]['se'] is not None else None for key in pooled}
        occz={key:(occ['probabilities'][key]-reference['probabilities'][key])/np.hypot(occse[key],reference['snis_mcse'][key]) for key in CENTRAL}
        ds={'mu':diagnostics(mm),'maximum':diagnostics(maximum),'tail6':diagnostics((maximum>6).astype(float))}
        screens={'summary':all(v is not None and abs(v)<=cfg['difference_mcse_max'] for v in z.values()),
            'chambers':all(abs(v)<=cfg['difference_mcse_max'] for v in occz.values()),
            **{key:r['rhat'] is not None and r['rhat']<=cfg['rhat_max'] and min(r['ess_bulk'],r['ess_tail'])>=cfg['ess_min'] for key,r in ds.items()}}
        results.append({'k':k,'reference':reference,'summary':pooled,'mcse':se,'difference_mcse':z,'occupancy':occ,'chamber_mcse':occse,'chamber_difference_mcse':occz,
            'per_chain_occupancy':chainocc,'per_chain_summary':chainmeans,'transitions':[transition_analysis(b,k) for _,b in traces],'diagnostics':ds,'screens':screens,
            'seconds':sum(r['seconds'] for r in prov),'counts':dict(sum((Counter(r['counts']) for r in prov),Counter()))})
        save(out/'new_block_comparisons.json',results);save(out/'new_block_oracle_checks.json',checks);print('blocks',k,ds,screens,flush=True)
    if any('error' in r or r.get('released_mle_residual',0)>1e-10 for r in checks):raise ArithmeticError('new selected sample oracle check failed')


def excursions(cfg,raw,out):
    rows=[];checks=[]
    for k in cfg['k_values']:
        for length in cfg['excursion_lengths']:
            x=math.sqrt(k)*np.array(geo.rational_witness(((0,0,0),(0,0,0))),dtype=float)
            state=block.selected_state(x,0.,k,mle_tolerance=1e-10);rng=np.random.default_rng(cfg['excursion_seed']+k*100+length)
            counts=Counter();examples=[]
            for it in range(cfg['excursion_blocks']):
                seed=cfg['excursion_seed']+k*100000+length*1000+it
                block.seed_score_rng(seed);trace=geo.traced_score_block(x,0.,0.,k,cfg['sigma_z'],length)
                inner=[geo.component(geo.chamber(v,k)) for v in trace[5][1:]]
                block.seed_score_rng(seed);new,state,d=block.block_step(x,0.,0.,k,cfg['sigma_z'],length,rng,state)
                if not np.array_equal(trace[0],d['proposed_x']):raise ArithmeticError('diagnostic replay changed proposal')
                invalid=any(v!='C0' for v in inner);counts['blocks']+=1;counts['accepted']+=int(d['accepted'])
                counts['inner_tail_steps']+=sum(v!='C0' for v in inner);counts['blocks_with_tail_excursion']+=int(invalid)
                counts['accepted_after_tail_excursion']+=int(d['accepted'] and invalid)
                counts['invalid_endpoints']+=int(inner[-1]!='C0');counts[d['rejection'] or 'accepted_endpoint']+=1
                counts['selector_calls']+=int(d['selector_called'])
                if d['accepted'] and invalid and len(examples)<2:
                    record={'iteration':it,'initial_x':x.tolist(),'inner_x':trace[5][1:].tolist(),'inner_components':inner,'pairs':trace[6].tolist()};examples.append(record)
                    idx=next(i for i,c in enumerate(inner) if c!='C0')
                    checks.append(oracle_check(trace[5][idx+1],case(k),f'excursion_k{k}_m{length}',it))
                    checks.append(oracle_check(new,case(k),f'excursion_k{k}_m{length}_accepted',it,normalized=True))
                x=new
            rows.append({'k':k,'block_length':length,'fixed_mu':0.,'counts':dict(counts),'examples':examples,'qualification':'mechanism-only fixed-mu diagnostic, not a posterior-reference chain; replay leaves the production proposal unchanged'})
            save(out/'block_excursions.json',rows);save(out/'excursion_oracle_checks.json',checks)
            print('excursions',k,length,dict(counts),flush=True)


def figures(cfg,raw,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'];components=['C0','T1-','T1+','T2-','T2+','T3-','T3+']
    grid=np.linspace(-1,1,161);u1,u2=np.meshgrid(grid,grid);u3=-u1-u2
    fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    hexagon=np.array([[-1,0],[-1,1],[0,1],[1,0],[1,-1],[0,-1],[-1,0]])
    for ax,bits in zip(axes.ravel(),itertools.product((0,1),repeat=3)):
        vals=np.full(u1.shape,np.nan)
        for i,j in np.ndindex(u1.shape):
            u=[u1[i,j],u2[i,j],u3[i,j]]
            if max(map(abs,u))>1 or any(b and abs(v)<1e-10 for b,v in zip(bits,u)):continue
            node=(bits,tuple(int(np.sign(v)) if b else 0 for v,b in zip(u,bits)))
            vals[i,j]=components.index(geo.component(node))
        ax.pcolormesh(u1,u2,vals,cmap=ListedColormap(colors),vmin=-.5,vmax=6.5,rasterized=True,shading='auto')
        ax.plot(*hexagon.T,color='black',lw=1.3);ax.axvline(0,color='white',lw=.8,ls=':');ax.axhline(0,color='white',lw=.8,ls=':');ax.plot([-1,1],[1,-1],color='white',lw=.8,ls=':')
        ax.set_aspect('equal');ax.set_title(''.join(map(str,bits)));ax.set_xlabel('u1');ax.set_ylabel('u2')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=n) for c,n in zip(colors,components)],loc='outside lower center',ncol=7)
    fig.suptitle('Exact n=3 classification over the score hexagon; green = selected-MLE support\nInner/outer sheet bits; u=2 sqrt(k) z. Outer-coordinate axes are infinite ends.')
    fig.savefig(out/'score_sheets.png',dpi=160);fig.savefig(out/'score_sheets.pdf');plt.close(fig)
    graph=chamber_graph();fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True);positions={}
    for node in chambers():
        if sum(node[0])==0:positions[node]=(0,0)
    # The six one-outer chambers form a cycle after subdividing each edge
    # with its opposite-sign two-outer chamber. Draw that wheel without crossings.
    central_one=[((1,0,0),(1,0,0)),((0,1,0),(0,-1,0)),((0,0,1),(0,0,1)),
        ((1,0,0),(-1,0,0)),((0,1,0),(0,1,0)),((0,0,1),(0,0,-1))]
    for j,n in enumerate(central_one):
        theta=2*math.pi*j/6;positions[n]=(2.*math.cos(theta),2.*math.sin(theta))
        neighbor=central_one[(j+1)%6]
        common=next(v for v in graph[n]&graph[neighbor] if sum(v[0])==2)
        theta+=math.pi/6;positions[common]=(2.*math.cos(theta),2.*math.sin(theta))
    for i,c in enumerate(components[1:]):
        nodes=sorted(n for n in chambers() if geo.component(n)==c)
        for j,n in enumerate(nodes):positions[n]=(4.+2*j,2.4-i*.95)
    for n,vs in graph.items():
        for v in vs:
            if geo.node_name(n)<geo.node_name(v):ax.plot([positions[n][0],positions[v][0]],[positions[n][1],positions[v][1]],color='#555555',lw=1)
    for n,xy in positions.items():
        ax.scatter(*xy,s=350,color=colors[components.index(geo.component(n))],zorder=3,edgecolor='white');ax.annotate(geo.node_name(n),xy,xytext=(0,13),textcoords='offset points',ha='center',fontsize=8)
    ax.text(0,-2.7,'C0: 13 chambers, all selected-MLE valid',ha='center');ax.text(5,-3.1,'Six disjoint two-chamber tail components; all invalid',ha='center');ax.set_title('25 chambers, 24 finite-fold edges, seven finite connected components');ax.axis('off')
    fig.savefig(out/'component_graph.png',dpi=160);fig.savefig(out/'component_graph.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--phase',choices=['algebra','saved','stress','blocks','excursions','figures'],required=True);a=ap.parse_args()
    cfg=load(a.config);raw=ROOT/cfg['raw_directory'];out=ROOT/cfg['evidence_directory'];raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    sources=['models/loc/student/score_gibbs.py','models/loc/student/selected_gibbs.py','models/loc/student/selected_block.py','models/loc/student/selected_rattle.py','models/loc/student/mle.py','reporting/diagnostics/student_n3_selected_geometry.py','scripts/study_student_n3_selected_geometry.py']
    save(out/(a.phase+'_execution.json'),{'source_sha256':{p:sha(ROOT/p) for p in sources},'config_sha256':sha(a.config),'python':sys.version,'phase':a.phase,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    globals()[a.phase](cfg,raw,out)
if __name__=='__main__':main()
