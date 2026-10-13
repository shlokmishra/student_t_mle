"""Independent target, numerical, reversibility and failure checks for location kernels."""
from types import SimpleNamespace
from fractions import Fraction
import math
import numpy as np
import pytest
from scipy import integrate, stats, optimize
from kde_ref.moments import gaussian_kde_gaussian_prior_moments
from models.shared.selected_numerics import mh_accept, truncnorm_log_mass
from models.loc.student import selected_block as sb, selected_rattle as sr
from models.loc.student.endpoint import endpoint_certificate, score_bounds
from models.loc.student.mle import global_mle
from models.loc.logistic import selected as log, selected_gibbs as lg, selected_rattle as lr
from models.loc.laplace import selected as lap, selected_gibbs as pg, selected_facet_hmc as ph


def params(n=3,t=0.):
    return dict(n=n,initial_x=t+np.linspace(-.6,.6,n),num_iterations_T=40,initial_mu=.1,
                prior_mean=-.2,prior_std=1.3,proposal_std_mu=.3,proposal_std_z=.1,
                block_length=4,k=2.,b=1.,rattle_step_size=.03,rattle_num_steps=4)


@pytest.mark.parametrize('z,h,t,pm,ps', [([-1.,1.],1.2,0.,0.,2.),([-2.,.3,1.],.2,1.25,-.75,.6),([0.,.01],1e-4,.2,0.,1.)])
def test_kde_mixture_against_independent_quadrature(z,h,t,pm,ps):
    z=np.array(z);b=SimpleNamespace(samples=z,bandwidth=h)
    out=gaussian_kde_gaussian_prior_moments(b,t,pm,ps)
    def density(mu):return stats.norm.pdf(mu,pm,ps)*np.mean(stats.norm.pdf(t-mu,z,h))
    # Integrate each known component window, with its center as a quadrature point.
    points=sorted(set([pm-12*ps,pm+12*ps]+[t-v+d*h for v in z for d in [-12,0,12]]))
    def integral(fun,upper=np.inf):
        return sum(integrate.quad(lambda x:fun(x)*density(x),a,min(b,upper),epsabs=1e-12)[0]
                   for a,b in zip(points[:-1],points[1:]) if a<upper)
    norm=integral(lambda x:1.)
    np.testing.assert_allclose(out['normalization_constant'],norm,rtol=1e-8)
    np.testing.assert_allclose(out['posterior_mean'],integral(lambda x:x)/norm,rtol=1e-8,atol=1e-10)
    for key,p in [('posterior_q025',.025),('posterior_q50',.5),('posterior_q975',.975)]:
        assert abs(integral(lambda x:1.,out[key])/norm-p)<1e-7


def test_kde_recorded_counterexample_and_log_underflow():
    kde=stats.gaussian_kde([-1.,1.]);b=SimpleNamespace(samples=np.array([-1.,1.]),bandwidth=np.sqrt(kde.covariance[0,0]))
    r=gaussian_kde_gaussian_prior_moments(b,0.,0.,2.)
    np.testing.assert_allclose([r['posterior_q025'],r['posterior_q50'],r['posterior_q975']],[-2.461854799720723,0.,2.4618547997207236],atol=1e-10)
    assert abs(r['normalization_constant']-.1551458908062587)<1e-12
    r=gaussian_kde_gaussian_prior_moments(b,1000.,0.,.1)
    assert r['normalization_constant']==0 and np.isfinite(r['log_normalization_constant'])
    assert gaussian_kde_gaussian_prior_moments(object(),0.) is None
    for b in [SimpleNamespace(samples=[],bandwidth=1.),SimpleNamespace(samples=[np.nan],bandwidth=1.),SimpleNamespace(samples=[1.],bandwidth=0.)]:
        with pytest.raises(ValueError):gaussian_kde_gaussian_prior_moments(b,0.)


def test_unfloored_mh_and_narrow_normal_mass():
    assert mh_accept(-100.,0.) and mh_accept(-100.,1e-50)
    assert not mh_accept(-100.,.5)
    with pytest.raises(FloatingPointError):mh_accept(np.nan,.5)
    with pytest.raises(ValueError):mh_accept(0.,1.)
    np.testing.assert_allclose(truncnorm_log_mass(0.,1.,-1e-18,1e-18),np.log(2e-18/np.sqrt(2*np.pi)),rtol=1e-14)


@pytest.mark.parametrize('n',[4,5,10])
def test_student_cluster_root_certificate(n):
    for k in [1.,2.,3.]:
        t=1.25;x=t+np.linspace(-.3,.3,n);g=sr.geometry(x,t,k)
        c=endpoint_certificate(x,t,k,1e-10,g)
        assert c.status==1 and abs(global_mle(x,k)-t)<1e-12
        lo,hi=score_bounds(x,t,k)
        exact=sum((Fraction(float(a))-Fraction(t))/(Fraction(k)+(Fraction(float(a))-Fraction(t))**2) for a in x)
        assert Fraction(lo)<=exact<=Fraction(hi)
        y=np.array([-5.,-5.,0.,5.,5.]);gy=sr.geometry(y,0.,1.)
        assert endpoint_certificate(y,0.,1.,1e-10,gy).status==0


def test_student_fast_and_selector_chains_agree():
    p=params();p['initial_x']=np.array([-.4,0.,.4])
    for run in [sb.run_gibbs,sr.run_rattle]:
        a=run(934,0.,{**p,'mle_check':'auto'},False);b=run(934,0.,{**p,'mle_check':'selector'},False)
        np.testing.assert_array_equal(a['x_chain'],b['x_chain']);np.testing.assert_array_equal(a['mu_chain'],b['mu_chain'])
        assert a['counts']['selector_calls']<b['counts']['selector_calls']
        with pytest.raises(ValueError):run(1,0.,{**p,'mle_check':'auto','n3_endpoint_fast_path':True},False)


def test_logistic_n2_target_curvature_and_pair_flux():
    for a in [.1,1.,3.]:
        x=np.array([-a,a]);c,j,g,_,_=log.geometry(x,0.)
        assert abs(c)<1e-14
        # Relative pair-score densities: q(z)^2*J dz equals f(a)f(-a) da.
        z=math.tanh(a/2);jac=.5*(1-z*z)
        value=np.exp(2*log.q_logpdf(z,0.))*j*jac
        assert abs(value/(2*np.exp(log.loglik(x,0.)))-1)<1e-12
    z,zp,delta,mu,sigma=.1,.23,.3,.4,.2
    ratio=log.pair_log_ratio(z,zp,delta,mu,sigma)
    rev=log.pair_log_ratio(zp,z,delta,mu,sigma)
    assert abs(ratio+rev)<1e-13
    lo,hi=-.7,1.
    target=lambda q:np.exp(log.q_logpdf(q,mu)+log.q_logpdf(delta-q,mu))
    h=lambda b,a:stats.norm.pdf(b,a,sigma)/np.exp(truncnorm_log_mass(a,sigma,lo,hi))
    np.testing.assert_allclose(target(z)*h(zp,z)*min(1,np.exp(ratio)),target(zp)*h(z,zp)*min(1,np.exp(rev)),rtol=1e-12)


def test_logistic_force_and_phase_reversal_volume():
    x=np.array([-.7,.1,.8]);mu=.2
    _,f=log.potential_force(x,mu,0.)
    for i in range(3):
        d=np.zeros(3);d[i]=1e-6
        numerical=(log.potential_force(x+d,mu,0.)[0]-log.potential_force(x-d,mu,0.)[0])/2e-6
        assert abs(numerical-f[i])<1e-8
    def phase(v):
        q=np.array([v[0],-v[0]])/np.sqrt(2);p=np.array([v[1],-v[1]])/np.sqrt(2)
        qp,pp,ok=lr.phase_proposal(q,p,.3,0.,.04,5);assert ok
        return np.array([qp[0],pp[0]])*np.sqrt(2)
    v=np.array([.8,.4]);out=phase(v);np.testing.assert_allclose(phase(out),v,atol=1e-10)
    jac=np.column_stack([(phase(v+np.eye(2)[i]*1e-6)-phase(v-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
    assert abs(abs(np.linalg.det(jac))-1)<1e-6


@pytest.mark.parametrize('mu,upper',[(-2.,1.),(2.,-1.),(0.,.2)])
def test_laplace_truncation_against_conditional_cdf(mu,upper):
    sb.seed_score_rng(281);x=np.array([lap.sample_left(mu,upper,1.) for _ in range(3000)])
    transformed=stats.laplace.cdf(x,mu)/stats.laplace.cdf(upper,mu)
    assert stats.kstest(transformed,'uniform').statistic<.035
    assert np.all(x<upper)
    assert lap.sample_left(1000.,0.,1.)<0.
    assert lap.sample_right(-1000.,0.,1.)>0.


@pytest.mark.parametrize('mu,upper',[(0.,np.inf),(1.,3.),(4.,2.)])
def test_laplace_gap_conditional(mu,upper):
    sb.seed_score_rng(284);x=np.array([lap.draw_middle_gap(mu,0.,1.,upper) for _ in range(3000)])
    end=upper if np.isfinite(upper) else 40.
    norm=integrate.quad(lambda d:np.exp(-2*max(d,abs(mu))),0,end,points=[abs(mu)] if 0<abs(mu)<end else None)[0]
    for value in [.2,.8,1.5]:
        if value>=upper:continue
        probability=integrate.quad(lambda d:np.exp(-2*max(d,abs(mu))),0,value)[0]/norm
        assert abs(np.mean(x<=value)-probability)<.035


@pytest.mark.parametrize('n',[2,3,4,5])
def test_laplace_reflection_reversibility_constraints(n):
    q=np.linspace(-.6,.6,n);p=lap.tangent(np.random.default_rng(n).normal(size=n))
    new,pp,ok,events,error=ph.phase_proposal(q,p,.2,1.,.06,8)
    assert ok and error<1e-10
    assert lap.check_state(new,0.)<1e-10
    np.testing.assert_allclose(lap.tangent(pp),pp,atol=1e-13)
    qd,pd,e=lap.reflected_drift(q,p,2.)
    qb,pb,_=lap.reflected_drift(qd,-pd,2.)
    np.testing.assert_allclose(qb,q,atol=1e-12);np.testing.assert_allclose(pb,-p,atol=1e-12)
    assert abs(np.dot(pd,pd)-np.dot(p,p))<1e-12


def test_laplace_reflection_volume_and_ambiguous_failure():
    def phase(v):
        q=np.array([-v[0],v[0]])/np.sqrt(2);p=np.array([-v[1],v[1]])/np.sqrt(2)
        q,p,_,_,_=ph.phase_proposal(q,p,.1,1.,.1,3)
        return np.array([q[1],p[1]])*np.sqrt(2)
    v=np.array([.2,-1.2]);jac=np.column_stack([(phase(v+np.eye(2)[i]*1e-6)-phase(v-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
    assert abs(abs(np.linalg.det(jac))-1)<1e-6
    with pytest.raises(FloatingPointError,match='simultaneous'):
        lap.reflected_drift(np.array([-1.,0.,1.]),np.array([1.,0.,-1.]),2.)


@pytest.mark.parametrize('module,name,n',[(lg,'run_gibbs',3),(lg,'run_gibbs',10),(lr,'run_rattle',3),(lr,'run_rattle',10),(pg,'run_gibbs',3),(pg,'run_gibbs',4),(ph,'run_hmc',3),(ph,'run_hmc',4)])
def test_new_chains_reproducible_independent_and_valid(module,name,n):
    t=1.25;p=params(n,t);run=getattr(module,name)
    old=np.random.get_state();a=run(291,t,p,False);b=run(291,t,p,False);c=run(292,t,p,False)
    np.testing.assert_array_equal(a['x_chain'],b['x_chain']);np.testing.assert_array_equal(a['mu_chain'],b['mu_chain']);assert not np.array_equal(a['mu_chain'],c['mu_chain'])
    np.testing.assert_array_equal(old[1],np.random.get_state()[1]);assert np.isfinite(a['x_chain']).all()
    if 'logistic' in module.__name__:
        for x in a['x_chain']:
            root=optimize.brentq(lambda v:np.tanh((x-v)/2).sum(),x.min(),x.max(),xtol=1e-13)
            assert abs(root-t)<1e-10
    else:
        np.testing.assert_allclose(np.median(a['x_chain'],axis=1),t,atol=1e-10)
        assert len(set(np.argmin(abs(a['x_chain'][1:]-t),axis=1)))>1
    for bad in [dict(prior_std=-1),dict(initial_x=np.ones(n)*np.nan)]:
        with pytest.raises(ValueError):run(1,t,{**p,**bad},False)


def test_new_failure_context_and_no_silent_stay(monkeypatch):
    def fail(*args):raise FloatingPointError('injected failure')
    monkeypatch.setattr(lg,'score_block',fail)
    with pytest.raises(FloatingPointError) as e:lg.run_gibbs(2,0.,params(),False)
    assert e.value.iteration==1 and e.value.partial['x_chain'].shape==(1,3)
    with pytest.raises(FloatingPointError):log.check_release(np.array([-.4,0.,.4])+1e-8,0.)


def test_logistic_curved_phase_canonical_volume():
    # Graph chart q=(x1,x2); r=B.T p are canonical cotangent coordinates.
    def chart(q):
        x=np.r_[q,-2*np.arctanh(np.tanh(q/2).sum())]
        normal=log.geometry(x,0.)[-1]
        b=np.vstack([np.eye(2),-normal[:2]/normal[2]])
        return x,b
    def phase(v):
        x,b=chart(v[:2]);p=b@np.linalg.solve(b.T@b,v[2:])
        xx,pp,ok=lr.phase_proposal(x,p,.3,0.,.04,4)
        assert ok and abs(log.geometry(xx,0.)[0])<1e-12
        _,bb=chart(xx[:2])
        return np.r_[xx[:2],bb.T@pp]
    v=np.array([-.4,.6,.3,-.2])
    np.testing.assert_allclose(phase(phase(v)),v,atol=1e-10)
    jac=np.column_stack([(phase(v+np.eye(4)[i]*1e-6)-phase(v-np.eye(4)[i]*1e-6))/2e-6 for i in range(4)])
    assert abs(abs(np.linalg.det(jac))-1)<2e-6


def test_laplace_even_reflection_canonical_volume():
    basis=np.array([[1.,0.,0.],[0.,-1/np.sqrt(2),0.],[0.,1/np.sqrt(2),0.],[0.,0.,1.]])
    def phase(v):
        q,p,ok,events,error=ph.phase_proposal(basis@v[:3],basis@v[3:],.1,1.,.1,5)
        assert ok
        return np.r_[basis.T@q,basis.T@p]
    v=np.array([-1.,.2,1.2,1.2,-.5,-.3])
    np.testing.assert_allclose(phase(phase(v)),v,atol=1e-10)
    jac=np.column_stack([(phase(v+np.eye(6)[i]*1e-6)-phase(v-np.eye(6)[i]*1e-6))/2e-6 for i in range(6)])
    assert abs(abs(np.linalg.det(jac))-1)<2e-6


def test_basu_sufficiency_counterexamples_and_gaussian_control():
    # Same selected T=0 but likelihood ratios depend on mu: T is not sufficient.
    def student_product(a,mu):
        return math.prod(Fraction(1)+(x-mu)**2 for x in [-a,0,a])
    assert student_product(1,0)/student_product(2,0)!=student_product(1,1)/student_product(2,1)
    # Logistic x=(-log(a),0,log(a)), exp(mu)=m: density is u*m/(u+m)^2.
    def logistic_product(a,m):
        return math.prod(u*m/(u+m)**2 for u in [1/Fraction(a),Fraction(1),Fraction(a)])
    assert logistic_product(2,1)/logistic_product(4,1)!=logistic_product(2,2)/logistic_product(4,2)
    def laplace_sum(a,mu): return sum(abs(x-mu) for x in [-a,0,a])
    assert laplace_sum(1,0)-laplace_sum(2,0)!=laplace_sum(1,Fraction(3,2))-laplace_sum(2,Fraction(3,2))
    # Gaussian mean and centered residual are jointly normal and uncorrelated.
    n=5;projection=np.eye(n)-np.ones((n,n))/n
    np.testing.assert_allclose(np.ones(n)/n@projection,np.zeros(n),atol=1e-15)
    t,pm,ps=1.25,-.75,.6
    var=1/(n+1/ps**2);mean=var*(n*t+pm/ps**2)
    density=lambda mu:stats.norm.pdf(t,mu,1/np.sqrt(n))*stats.norm.pdf(mu,pm,ps)
    norm=integrate.quad(density,-np.inf,np.inf)[0]
    np.testing.assert_allclose(integrate.quad(lambda mu:mu*density(mu),-np.inf,np.inf)[0]/norm,mean,rtol=1e-10)
    np.testing.assert_allclose(integrate.quad(lambda mu:(mu-mean)**2*density(mu),-np.inf,np.inf)[0]/norm,var,rtol=1e-10)


def test_selected_registry_is_explicit():
    import importlib
    from models.model_registry import SELECTED_LOCATION_METHODS
    assert len(SELECTED_LOCATION_METHODS)==7
    for row in SELECTED_LOCATION_METHODS.values():
        assert callable(getattr(importlib.import_module(row['module']),row['entrypoint']))


def test_laplace_saved_contact_roundoff_regression():
    # First smoke stopped at iteration 8 with a tiny negative post-contact gap.
    p={**params(), 'initial_x':np.array([.3254433512125686,0.,-.4774885897683452]),
       'initial_mu':-.2383685306588924, 'prior_mean':0., 'prior_std':2.,
       'proposal_std_mu':.45, 'rattle_step_size':.05, 'rattle_num_steps':10,
       'num_iterations_T':32}
    r=ph.run_hmc(2026102277,0.,p,False)
    assert r['counts']['wall_events']>0 and r['max_reverse_error']<1e-10
    np.testing.assert_array_equal(np.median(r['x_chain'],axis=1),np.zeros(33))


def test_archived_source_is_not_current_equivalence():
    import hashlib
    from pathlib import Path
    from diagnostics.source_provenance import verify_archived_source
    import json
    root=Path(__file__).resolve().parents[1]
    before=json.loads((root/'docs/research/evidence/location_only_correctness_2026-10-12/source_before.json').read_text())
    path='models/loc/student/selected_block.py'
    r=verify_archived_source(root,path,before[path])
    assert r['status']=='verified_archived_source_only' and r['current_numerical_equivalence'] is False
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()!=before[path]
