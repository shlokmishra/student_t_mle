"""Exact binary-input intervals, archived rational oracle, and paired kernels."""
from fractions import Fraction as F
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from models.loc.student import selected_block as block
from models.loc.student import selected_rattle as support
from models.loc.student.mle import global_mle, StudentMLEError
from models.loc.student.n3_endpoint import certify_endpoint, score_interval

ROOT=Path(__file__).resolve().parents[1]
STRESS=ROOT/'docs/research/evidence/student_n3_geometry_selected_mle_2026-10-10/stress_oracle.json'


def quantities(x,t,k):
    y=np.asarray(x)-t; d=k+y*y
    return float(np.sum((k-y*y)/(d*d))),float(np.sum(y/d))


@pytest.mark.parametrize('k',[1.,2.,3.])
def test_outward_score_contains_exact_rational_binary_score(k):
    rng=np.random.default_rng(9183)
    datasets=[rng.normal(size=3)*10**s for s in [-100,-5,0,5,20,100]]
    datasets += [np.array([0.,np.nextafter(0.,1.),-np.nextafter(0.,1.)]),np.array([-1.,1.,1e20])]
    for x in datasets:
        for t in [0.,.75,np.nextafter(1.,2.)]:
            lo,hi=score_interval(x,t,k)
            score=sum((F(float(v))-F(t))/(F(k)+(F(float(v))-F(t))**2) for v in x)
            assert F(lo)<=score<=F(hi)


@pytest.mark.parametrize('k',[1,2,3])
def test_all_25_chambers_against_saved_exact_oracle_and_current_selector(k):
    rows=[r for r in json.loads(STRESS.read_text()) if r['k']==k and r['source'].startswith('rational_chamber')]
    assert len(rows)==25
    for r in rows:
        x=np.asarray(r['data']);j,c=quantities(x,0.,k)
        cert=certify_endpoint(x,0.,k,1e-10,j,c)
        selected=global_mle(x,k)
        assert abs(selected-r['oracle']['selected'])<1e-10
        if r['predicted_component']=='C0':
            assert cert.status==1
            a,b=map(F,r['oracle']['selected_interval'])
            assert F(cert.lower)<=a<=b<=F(cert.upper)
        else:
            assert cert.status!=1
            assert abs(selected)>1e-10


def test_archived_adversaries_never_falsely_accept():
    for r in json.loads(STRESS.read_text()):
        if r['source'].startswith('rational_chamber'):continue
        x=np.asarray(r['data']);k=r['k'];j,c=quantities(x,0.,k)
        cert=certify_endpoint(x,0.,k,1e-10,j,c)
        if cert.status==1:
            a,b=map(F,r['oracle']['selected_interval'])
            assert F(cert.lower)<=a<=b<=F(cert.upper)
        if abs(r['oracle']['selected'])>1e-10:assert cert.status!=1


@pytest.mark.parametrize('k',[1.,2.,3.])
@pytest.mark.parametrize('t',[0.,1.25,-.75])
def test_tiny_residual_requires_root_bound_not_small_score(k,t):
    tol=1e-10
    for eps in [0.,tol/2,2*tol,1e-8]:
        x=np.full(3,t+eps);j,c=quantities(x,t,k)
        cert=certify_endpoint(x,t,k,tol,j,c)
        assert (cert.status==1)==(abs(float(x[0])-t)<tol)
        if cert.status==1:
            assert F(cert.lower)<=F(float(x[0]))<=F(cert.upper)
            assert max(F(t)-F(cert.lower),F(cert.upper)-F(t))<=F(tol)


@pytest.mark.parametrize('k',[1.,2.,3.])
def test_fold_large_offset_extremes_and_flat_states_fall_back(k):
    x=np.array([-np.sqrt(k),0.,np.sqrt(k)])
    for eps in [0.,1e-10]:
        y=x.copy();y[0]+=eps;j,c=quantities(y,0.,k)
        assert certify_endpoint(y,0.,k,1e-10,j,c).reason=='fold_adjacent'
    assert certify_endpoint(np.full(3,2.**40),2.**40,k,1e-10,3/k,0.).reason=='unrepresentable_bracket'
    assert certify_endpoint(np.array([0.,.25,1e200]),0.,k,1e-10,1.,0.).reason=='interval_unresolved'
    # Tiny score at a flat, non-selected root cannot be promoted to certification.
    assert certify_endpoint(np.array([-1.,1.,1e20]),0.,1.,1e-10,1e-30,1e-20).status==0


def test_no_rounded_tolerance_window_false_rejection():
    # Endpoint tests use an OUTWARD release window for exclusion, inward for acceptance.
    t=1.25;tol=1e-10
    for eps in [np.nextafter(tol,0.),tol,np.nextafter(tol,np.inf)]:
        x=np.full(3,t+eps);j,c=quantities(x,t,2.)
        cert=certify_endpoint(x,t,2.,tol,j,c)
        if abs(F(float(x[0]))-F(t))<=F(tol):assert cert.status!=-1


def test_translation_permutation_and_signs():
    for t in [0.,1.25,-.75]:
        for p in itertools.permutations([-.3,0.,.3]):
            x=t+np.array(p);j,c=quantities(x,t,2.)
            assert certify_endpoint(x,t,2.,1e-10,j,c).status==1


@pytest.mark.parametrize('n,t,k,m',[(3,0.,1.,4),(3,1.25,2.,16),(3,-.75,3.,4),(10,0.,2.,4)])
def test_old_and_guarded_chain_identical_rng_and_retained_states(n,t,k,m):
    e=np.random.default_rng(239+n).standard_t(k,n);x=t+e-global_mle(e,k)
    p=dict(n=n,k=k,num_iterations_T=150,initial_x=x,initial_mu=.2,prior_mean=0.,prior_std=2.,proposal_std_mu=.45,proposal_std_z=.1,block_length=m)
    old=block.run_gibbs(838,t,{**p,'mle_check':'selector'},False)
    new=block.run_gibbs(838,t,{**p,'n3_endpoint_fast_path':True},False)
    np.testing.assert_array_equal(old['mu_chain'],new['mu_chain'])
    np.testing.assert_array_equal(old['x_chain'],new['x_chain'])
    for name in ('accepted','score_accepts','branch_changes_accepted','global_selection','cheap_metropolis'):
        assert old['counts'][name]==new['counts'][name]
    if n>3:
        assert new['counts']['n3_fast_attempts']==0
        assert old['counts']['selector_calls']==new['counts']['selector_calls']
    else:assert new['counts']['selector_calls']<old['counts']['selector_calls']


def test_ambiguous_certificate_calls_original_and_propagates_failure(monkeypatch):
    x=np.array([-.3,0.,.3]);state=support.selected_state(x,0.,2.)
    y=np.array([-np.sqrt(2.),0.,np.sqrt(2.)])
    monkeypatch.setattr(block,'score_block_kernel',lambda *a:(y,1,1,1,0.))
    def fail(*a,**kw):raise StudentMLEError('selection_unresolved','injected fallback')
    monkeypatch.setattr(block,'selected_state',fail)
    class Zero:
        def random(self):return 0.
    with pytest.raises(StudentMLEError):
        block.block_step(x,0.,0.,2.,.1,4,Zero(),state,n3_endpoint_fast_path=True)


def test_bad_inputs():
    for x in [[1.,2.],[1.,2.,np.inf],[1j,2.,3.]]:
        with pytest.raises(ValueError):certify_endpoint(x,0.,2.,1e-10,1.,0.)
    for tol in [0.,-1.,np.nan]:
        with pytest.raises(ValueError):certify_endpoint(np.zeros(3),0.,2.,tol,1.,0.)
