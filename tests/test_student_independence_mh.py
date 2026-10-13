"""Independent MH flux, selected reconstruction, cache count and failure tests."""
import math
import numpy as np
import pytest

from models.loc.student import independence as ind
from models.loc.student.mle import global_mle, StudentMLEError


def params(n=10,t=0.):
    return dict(n=n,k=2.,prior_mean=-.75,prior_std=.6,num_iterations_T=120,
                initial_e=np.random.default_rng(12).standard_t(2.,n))


def test_independent_proposal_flux_with_unequal_base_probabilities():
    es=[np.array([-.5,0.,1.]),np.array([3.,1.,-2.]),np.array([-.2,.5,.75]),np.array([-4.,2.,8.])]
    states=[ind.make_state(e,1.25,2.,-.75,.6) for e in es]
    q=np.array([.1,.25,.45,.2]);w=np.exp([s.log_weight for s in states]);pi=q*w;pi/=pi.sum()
    kernel=np.empty((4,4))
    for i in range(4):
        for j in range(4):kernel[i,j]=q[j]*math.exp(ind.log_acceptance(states[i],states[j]))
        kernel[i,i]+=1-kernel[i].sum()
    np.testing.assert_allclose(pi[:,None]*kernel,(pi[:,None]*kernel).T,atol=2e-16)
    np.testing.assert_allclose(pi@kernel,pi,atol=2e-16)


@pytest.mark.parametrize('t,scale,k',[(0.,1.,2.),(1.25,1.,2.),(-.75,2.,3.)])
def test_selected_mle_reconstruction_and_prior_sign(t,scale,k):
    e=scale*np.array([-1.,.3,1.6,2.,0.])
    state=ind.make_state(e,t,k,-.75,.6,scale=scale)
    assert abs(global_mle(state.x/scale,k)*scale-t)<1e-10
    assert state.mu==t-state.selected_error_mle
    assert state.log_weight==-.5*((state.mu+.75)/.6)**2
    np.testing.assert_allclose(state.x-state.mu,e,atol=4e-16)
    assert not state.x.flags.writeable and not state.e.flags.writeable


def test_exact_global_tie_uses_selected_smaller_root():
    e=np.array([-2.,-2.,2.,2.])
    t=global_mle(e,1.)
    state=ind.make_state(e,t,1.,0.,2.)
    assert abs(state.selected_error_mle+math.sqrt(3))<1e-12
    assert abs(global_mle(state.x,1.)-t)<1e-12


def test_rounded_tied_reconstruction_fails_without_a_second_selector_call(monkeypatch):
    original=ind.global_mle;calls=[]
    def track(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(ind,'global_mle',track)
    with pytest.raises(FloatingPointError,match='nonuniform rounding'):
        ind.make_state(np.array([-2.,-2.,2.,2.]),1.25,1.,0.,2.)
    assert len(calls)==1


def test_one_global_selector_call_per_proposal_and_cached_rejection(monkeypatch):
    original=ind.global_mle;calls=[]
    def tracked(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(ind,'global_mle',tracked)
    r=ind.run_independence(324,1.25,params(),False)
    assert len(calls)==121==r['counts']['selector_calls']
    for i,accepted in enumerate(r['accepted_flags']):
        if not accepted:
            assert r['mu_chain'][i+1]==r['mu_chain'][i]
            np.testing.assert_array_equal(r['x_chain'][i+1],r['x_chain'][i])


def test_fixed_seeds_reproduce_independent_chains_without_global_rng_mutation():
    before=np.random.get_state();p=params()
    a=ind.run_independence(35,1.25,p,False);b=ind.run_independence(35,1.25,p,False);c=ind.run_independence(36,1.25,p,False)
    np.testing.assert_array_equal(a['x_chain'],b['x_chain']);np.testing.assert_array_equal(a['mu_chain'],b['mu_chain'])
    assert not np.array_equal(a['mu_chain'],c['mu_chain'])
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])


def test_selector_failure_aborts_with_partial_chain(monkeypatch):
    original=ind.global_mle;calls=[]
    def fail(*a,**kw):
        calls.append(1)
        if len(calls)==4:raise StudentMLEError('selection_unresolved','injected')
        return original(*a,**kw)
    monkeypatch.setattr(ind,'global_mle',fail)
    with pytest.raises(StudentMLEError) as e:ind.run_independence(112,0.,params(),False)
    assert e.value.iteration==3 and len(e.value.partial['mu_chain'])==3


def test_zero_uniform_and_rejection_rollback():
    state=ind.make_state(np.array([-.3,0.,.3]),0.,2.,0.,2.)
    class Fixed:
        def __init__(self,u):self.u=u
        def standard_t(self,*a):return np.array([10.,11.,12.])
        def random(self):return self.u
    new,accepted,candidate=ind.independence_step(state,Fixed(.9),0.,2.,0.,2.)
    assert not accepted and new is state
    new,accepted,candidate=ind.independence_step(state,Fixed(0.),0.,2.,0.,2.)
    assert accepted and new is candidate
    with pytest.raises(ValueError,match='different target'):ind.independence_step(state,Fixed(.5),1.,2.,0.,2.)


@pytest.mark.parametrize('bad',[[1.,2.],[1.,np.inf,3.],[1j,0.,1.]])
def test_bad_inputs(bad):
    p=params(3);p['initial_e']=bad
    with pytest.raises(ValueError):ind.run_independence(1,0.,p,False)


def test_nonfinite_weights_fail_explicitly():
    with pytest.raises((FloatingPointError,OverflowError)):ind.log_prior_weight(1e308,-1e308,.5)
