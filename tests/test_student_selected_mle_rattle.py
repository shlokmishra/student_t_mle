"""Independent energy, canonical-volume, reversal and failure checks."""
import math
import numpy as np
import pytest

from models.loc.student import score_rattle as legacy
from models.loc.student import selected_rattle as selected
from models.loc.student.selected_gibbs import inverse_branches
from models.loc.student.mle import global_mle, StudentMLEError


def initial(n=3,t=0.,scale=1.):
    e=np.random.default_rng(703+n).standard_t(2,n)
    return t+scale*(e-global_mle(e,2))


def params(x,**extra):
    return dict(n=len(x),k=2.,num_iterations_T=80,initial_x=x,initial_mu=.1,
                prior_mean=0.,prior_std=2.,proposal_std_mu=.45,
                rattle_step_size=.05,rattle_num_steps=4,**extra)


@pytest.mark.parametrize('n,t,s',[(3,0.,1.),(10,1.25,1.),(3,.75,2.)])
def test_curvature_gram_and_selected_state(n,t,s):
    x=initial(n,t,s); y=(x-t)/s; a=(2-y*y)/(2+y*y)**2
    g=selected.geometry(x,t,2,scale=s)
    np.testing.assert_allclose([g.curvature,g.gram],[a.sum(),a@a],rtol=3e-15)
    assert not np.isclose(g.curvature,math.sqrt(g.gram))
    state=selected.selected_state(x,t,2,scale=s)
    assert state is not None and abs(state.selected_mle-t)<1e-12


def test_endpoint_ratio_matches_full_extended_target_and_reverse():
    x0=np.array([-.4,.4,0.]); x1=initial()
    g0,g1=[selected.geometry(x,0.,2.) for x in [x0,x1]]
    p0=np.array([.1,-.2,.05]); p1=np.array([-.3,.2,.1]); mu=.31
    def independent_energy(x,p,g):
        return 1.5*np.log1p((x-mu)**2/2).sum()+.5*math.log(g.gram)+.5*(p@p)-g.log_j
    h0=legacy._hamiltonian(x0,p0,mu,2,0.);h1=legacy._hamiltonian(x1,p1,mu,2,0.)
    ratio=selected.endpoint_log_acceptance(h0,h1,g0,g1)
    np.testing.assert_allclose(ratio,independent_energy(x0,p0,g0)-independent_energy(x1,p1,g1),atol=2e-15)
    np.testing.assert_allclose(selected.endpoint_log_acceptance(h1,h0,g1,g0),-ratio,atol=2e-15)
    assert legacy._hamiltonian(x1,-p1,mu,2,0.)==h1
    # Direct detailed-balance flux for a volume-preserving involution.
    e0,e1=independent_energy(x0,p0,g0),independent_energy(x1,p1,g1)
    np.testing.assert_allclose(-e0+min(0.,ratio),-e1+min(0.,-ratio),atol=2e-15)


@pytest.mark.parametrize('n',[3,10])
def test_full_phase_reversal_constraints_and_shared_map(n):
    x=initial(n);opts=selected.options(params(x));rng=np.random.default_rng(731)
    p,ok=legacy._project_momentum_kernel(rng.normal(size=n),legacy._constraint_grad_kernel(x,0.,2.),opts['grad_tol']);assert ok
    xp,pp,valid,d=selected.phase_proposal(x,p,.2,2.,opts);assert valid
    xr,pr,validr,_=selected.phase_proposal(xp,pp,.2,2.,opts);assert validr
    np.testing.assert_allclose(xr,x,atol=2e-10);np.testing.assert_allclose(pr,p,atol=2e-10)
    assert abs(legacy._constraint_value_kernel(xp,0.,2.))<=opts['proj_tol']
    assert abs(legacy._constraint_grad_kernel(xp,0.,2.)@pp)<opts['tangent_tol']
    yl,pl,okl,*_=legacy._rattle_trajectory_kernel(x,p,.2,0.,2.,opts['step_size'],opts['num_steps'],
        opts['proj_tol'],opts['max_iters'],opts['grad_tol'],opts['damping'],opts['line_search'],opts['tangent_tol'])
    assert okl;np.testing.assert_array_equal(xp,yl);np.testing.assert_array_equal(pp,-pl)


@pytest.mark.parametrize('n',[3,5])
def test_shared_integrator_preserves_canonical_volume_and_symplectic_form(n):
    # Use a regular inner chart, with canonical momenta independent of the code.
    u=np.linspace(-.25,.25,n-1);pi=np.linspace(.11,-.08,n-1)
    opts=selected.options(dict(rattle_step_size=.025,rattle_num_steps=3,rattle_proj_tol=1e-14))
    def chart_map(v):
        body=v[:n-1]; zlast=-np.sum(body/(2+body*body))
        last=min(inverse_branches(zlast,2.),key=abs)
        x=np.r_[body,last];grad=(2-x*x)/(2+x*x)**2;a=grad[:-1]/grad[-1]
        pb=np.linalg.solve(np.eye(n-1)+np.outer(a,a),v[n-1:]);p=np.r_[pb,-a@pb]
        xp,pp,ok,_=selected.trajectory(x,p,.2,2.,opts);assert ok
        gp=(2-xp*xp)/(2+xp*xp)**2;ap=gp[:-1]/gp[-1]
        return np.r_[xp[:-1],pp[:-1]-ap*pp[-1]]
    v=np.r_[u,pi];h=3e-6
    jac=np.column_stack([(chart_map(v+h*np.eye(len(v))[i])-chart_map(v-h*np.eye(len(v))[i]))/(2*h) for i in range(len(v))])
    eye=np.eye(n-1);zero=np.zeros_like(eye);omega=np.block([[zero,eye],[-eye,zero]])
    np.testing.assert_allclose(jac.T@omega@jac,omega,atol=2e-7,rtol=0)
    np.testing.assert_allclose(np.linalg.det(jac),1.,atol=2e-7)


def fake_phase(proposed):
    def run(y,p,*args):
        return proposed.copy(),-p,True,dict(forward=None,reverse=None,reverse_position_error=0.,reverse_momentum_error=0.,rejection=None)
    return run


class Fixed:
    def __init__(self,u):self.u=u
    def standard_normal(self,n):return np.zeros(n)
    def random(self):return self.u


def test_cheap_uniform_rejection_skips_selector(monkeypatch):
    x=np.array([-.4,.4,0.]);state=selected.selected_state(x,0.,2.)
    proposed=initial();monkeypatch.setattr(selected,'phase_proposal',fake_phase(proposed))
    monkeypatch.setattr(selected,'endpoint_log_acceptance',lambda *a:-10.)
    def fail(*a,**kw):raise AssertionError('selector should be skipped')
    monkeypatch.setattr(selected,'global_mle',fail)
    xn,s,d=selected.rattle_step(x,.2,0.,2.,Fixed(.5),selected.options({}),current_state=state)
    assert d['rejection']=='cheap_metropolis' and not d['selector_called']
    np.testing.assert_array_equal(xn,x);assert s is state


def test_selected_invalid_endpoint_rejects_after_cheap_accept(monkeypatch):
    x=np.array([-.4,.4,0.]);state=selected.selected_state(x,0.,2.)
    proposed=np.array([inverse_branches(.05,2.)[1],inverse_branches(.05,2.)[1],min(inverse_branches(-.1,2.),key=abs)])
    assert selected.geometry(proposed,0.,2.).curvature>0
    assert global_mle(proposed,2.)>10
    monkeypatch.setattr(selected,'phase_proposal',fake_phase(proposed))
    xn,s,d=selected.rattle_step(x,.2,0.,2.,Fixed(0.),selected.options({}),current_state=state)
    assert d['rejection']=='global_selection' and d['selector_called']
    np.testing.assert_array_equal(xn,x);assert s is state


def test_unresolved_selection_is_not_an_ordinary_rejection(monkeypatch):
    x=np.array([-.4,.4,0.]);state=selected.selected_state(x,0.,2.)
    monkeypatch.setattr(selected,'phase_proposal',fake_phase(x))
    def fail(*a,**kw):raise StudentMLEError('selection_unresolved','injected')
    monkeypatch.setattr(selected,'global_mle',fail)
    with pytest.raises(StudentMLEError,match='selection_unresolved'):
        selected.rattle_step(x,.2,0.,2.,Fixed(0.),selected.options({}),current_state=state)


def test_wrong_reverse_momentum_rejects(monkeypatch):
    x=np.array([-.4,.4,0.]);p=np.array([.1,-.1,0.]);calls=[]
    def fake(y,m,*a):
        calls.append(1)
        d=dict(status=0,projection_iterations=0,position_failures=0,momentum_failures=0,max_score_residual=0.,completed_steps=1,max_tangent_residual=0.,min_transverse=1.,min_gram=.1)
        return x.copy(),(p.copy() if len(calls)==1 else -p+1.),True,d
    monkeypatch.setattr(selected,'trajectory',fake)
    _,_,ok,d=selected.phase_proposal(x,p,.2,2.,selected.options({}))
    assert not ok and d['reverse_position_error']==0 and d['reverse_momentum_error']>1


def test_failed_forward_projection_rejects_without_selection(monkeypatch):
    x=np.array([-.4,.4,0.]);state=selected.selected_state(x,0.,2.)
    opts=selected.options(dict(rattle_step_size=5.,rattle_num_steps=1,rattle_proj_max_iters=1))
    calls=[]
    def selector(*a,**kw):calls.append(1);return global_mle(*a,**kw)
    monkeypatch.setattr(selected,'global_mle',selector)
    # The bounded solver failure is proposal-domain rejection, not estimator failure.
    _,_,d=selected.rattle_step(x,0.,0.,2.,np.random.default_rng(391),opts,current_state=state)
    assert d['rejection'] in ['forward_projection','reverse_phase'] and not calls


def test_degenerate_curvature_and_bad_inputs_explicit():
    with pytest.raises(selected.SelectedRATTLENumericalError,match='gram_degenerate'):
        selected.geometry([-1.,-1.,1.,1.],0.,1.)
    a=math.sqrt(-5+4*math.sqrt(2))
    with pytest.raises(selected.SelectedRATTLENumericalError,match='curvature_unresolved'):
        selected.geometry([-a,a,-math.sqrt(3),math.sqrt(3)],0.,1.)
    with pytest.raises(selected.SelectedRATTLENumericalError,match='score_constraint'):
        selected.geometry([.1,.2,0.],0.,2.)
    with pytest.raises(ValueError): selected.geometry([0.,1j,2.],0.,2.)
    with pytest.raises(ValueError): selected.options(dict(reverse_check=False))
    with pytest.raises(ValueError): selected.options(dict(mass_matrix=np.eye(3)))
    with pytest.raises(selected.SelectedRATTLENumericalError,match='hamiltonian_nonfinite'):
        g=selected.geometry([-.4,.4,0.],0.,2.);selected.endpoint_log_acceptance(math.nan,0,g,g)


def test_endpoint_roundtrip_and_flat_constraint_failure_before_selection(monkeypatch):
    y=np.array([-.4,.4,0.])
    x,error=selected.reconstruct_endpoint(y,1.25,2.)
    np.testing.assert_allclose((x-1.25)/2.,y,atol=1e-15);assert error<1e-15
    with pytest.raises(selected.SelectedRATTLENumericalError,match='endpoint_reconstruction'):
        selected.reconstruct_endpoint(y,1e16,1.)
    from dataclasses import replace
    g=selected.geometry(y,0.,2.)
    g=replace(g,score=1e-12,curvature=1e-8)
    def no_selection(*a,**kw):raise AssertionError('must diagnose constraint accuracy first')
    monkeypatch.setattr(selected,'global_mle',no_selection)
    with pytest.raises(selected.SelectedRATTLENumericalError,match='constraint_location_accuracy'):
        selected.selected_state(y,0.,2.,endpoint_geometry=g)


def test_stale_cache_and_parameter_failure(monkeypatch):
    x=initial();state=selected.selected_state(x,0.,2.)
    with pytest.raises(ValueError,match='cached selected state'):
        selected.rattle_step(np.array([-.4,.4,0.]),0.,0.,2.,np.random.default_rng(44),selected.options({}),current_state=state)
    monkeypatch.setattr(selected,'unnormalized_mu_logpdf',lambda *args:math.nan)
    with pytest.raises(selected.SelectedRATTLENumericalError,match='parameter_density_nonfinite'):
        selected.run_rattle(19,0.,params(x),False)


@pytest.mark.parametrize('n,t,s',[(3,0.,1.),(10,0.,1.),(10,1.25,1.),(3,.75,2.)])
def test_chain_selected_constraint_reproducibility_independence(n,t,s):
    x=initial(n,t,s);p=params(x,known_scale=s)
    before=np.random.get_state();a=selected.run_rattle(913,t,p,False);b=selected.run_rattle(914,t,p,False);c=selected.run_rattle(913,t,p,False)
    np.testing.assert_array_equal(a['x_chain'],c['x_chain']);np.testing.assert_array_equal(a['mu_chain'],c['mu_chain'])
    assert not np.array_equal(a['mu_chain'],b['mu_chain'])
    after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]
    assert a['target']=='selected_global_student_location_mle'
    for xx in a['x_chain']:
        assert abs(legacy._constraint_value_kernel((xx-t)/s,0.,2.))<2e-12
        assert abs(s*global_mle(xx/s,2.)-t)<2e-10
    assert a['counts']['selector_calls']<=81
