"""Pair flux, self-adjoint block composition, endpoint correction and failures."""
import math

import numpy as np
import pytest
from scipy import stats

from models import loc_student as legacy
from models import loc_student_selected_mle as exact
from models import loc_student_selected_mle_block as block
from models import loc_student_selected_mle_rattle as support
from models.student_location_mle import StudentMLEError, global_mle


def initial(n=3,t=0.):
    e=np.random.default_rng(954+n).standard_t(2.,n)
    return t+e-global_mle(e,2.)


@pytest.mark.parametrize('k,z,loc',[(1.,.01,0.),(2.,-.2,.7),(3.,.25,-.4),(2.,0.,0.),(2.,1e-150,.3)])
def test_compiled_branch_law_matches_floor_free_machinery(k,z,loc):
    a,b,p,q=block.branch_law(z,loc,k);ys,ps,eq=exact.branch_distribution(z,loc,0.,k)
    np.testing.assert_allclose([a,b] if len(ys)==2 else [a],ys,rtol=3e-15,atol=0)
    np.testing.assert_allclose(p,ps[0],atol=4e-15);np.testing.assert_allclose(q,eq,atol=5e-13)


def test_legacy_strip_is_not_physical_support_and_new_pair_has_no_strip():
    b=1/(2*math.sqrt(2));z=b-5e-13
    assert legacy.q_logpdf_no_const(z,0.,0.,2.,*legacy.z_support(2.))==-math.inf
    assert math.isfinite(block.branch_law(z,0.,2.)[3])
    assert legacy.psi_inverse(0.,2.)==(0.,0.)
    assert block.branch_law(0.,0.,2.)[3]==math.log(2.)


def test_score_pair_joint_flux_and_branch_refresh_after_rejection():
    z,zp,delta,mu,sigma=.12,.05,.10,.35,.11
    b=1/(2*math.sqrt(2));lo,hi=max(-b,delta-b),min(b,delta+b)
    def h(a,v):return stats.truncnorm.logpdf(v,(lo-a)/sigma,(hi-a)/sigma,loc=a,scale=sigma)
    def w(y):return stats.t.logpdf(y-mu,df=2.)-math.log(abs((2-y*y)/(2+y*y)**2))
    def branch(z):
        ys=exact.inverse_branches(z,2.);lw=np.array([w(v) for v in ys]);q=np.logaddexp.reduce(lw)
        return lw,lw-q
    a,pa=branch(z);b,pb=branch(delta-z);c,pc=branch(zp);d,pd=branch(delta-zp)
    r=block.score_log_ratio(z,zp,delta,mu,2.,sigma)
    np.testing.assert_allclose(r,np.logaddexp.reduce(c)+np.logaddexp.reduce(d)-np.logaddexp.reduce(a)-np.logaddexp.reduce(b)+h(zp,z)-h(z,zp),atol=3e-13)
    for i in range(2):
        for j in range(2):
            for ii in range(2):
                for jj in range(2):
                    left=a[i]+b[j]+h(z,zp)+min(0.,r)+pc[ii]+pd[jj]
                    right=c[ii]+d[jj]+h(zp,z)+min(0.,-r)+pa[i]+pb[j]
                    np.testing.assert_allclose(left,right,atol=1e-12)
    # The diagonal-in-score refresh law also has symmetric joint flux.
    np.testing.assert_allclose(a[0]+b[0]+pa[1]+pb[1],a[1]+b[1]+pa[0]+pb[0],atol=2e-14)
    x=np.array([exact.inverse_branches(v,2.)[-1] for v in [.12,-.02,-.1]])
    block.seed_score_rng(178);found=False
    for _ in range(200):
        y=x.copy();accepted,_,changed,_=block.score_pair_kernel(y,0,1,mu,0.,2.,.1)
        if not accepted and changed:
            np.testing.assert_allclose(y[:2]/(2+y[:2]**2),x[:2]/(2+x[:2]**2),atol=2e-16);found=True;break
    assert found


def test_random_scan_powers_self_adjoint_and_arbitrary_composition_need_not_be():
    # Independent finite-state algebra: each proposal has its own conditional law.
    pi=np.array([.2,.3,.5]);root=np.sqrt(pi)
    k1=np.array([[.4,.6,0.],[.4,.6,0.],[0.,0.,1.]])
    k2=np.array([[1.,0.,0.],[0.,.375,.625],[0.,.375,.625]])
    for k in [k1,k2]:np.testing.assert_allclose(pi[:,None]*k,(pi[:,None]*k).T,atol=1e-15)
    assert not np.allclose(pi[:,None]*(k1@k2),(pi[:,None]*(k1@k2)).T)
    kernel=(k1+k2)/2
    for m in [1,4,16]:
        power=np.linalg.matrix_power(kernel,m);np.testing.assert_allclose(pi[:,None]*power,(pi[:,None]*power).T,atol=2e-15)
        symmetric=root[:,None]*power/root[None,:];np.testing.assert_allclose(symmetric,symmetric.T,atol=2e-15)


def test_outer_whole_block_rejection_retains_original_no_selector_on_cheap_fail(monkeypatch):
    x=initial();state=support.selected_state(x,0.,2.);proposed=x.copy();calls=[]
    monkeypatch.setattr(block,'score_block_kernel',lambda *a:(proposed,1,1,1,0.))
    from dataclasses import replace
    monkeypatch.setattr(block,'geometry',lambda *a,**kw:replace(state.geometry,log_j=state.geometry.log_j-100))
    monkeypatch.setattr(block,'selected_state',lambda *a,**kw:calls.append(1))
    y,s,d=block.block_step(x,.1,0.,2.,.1,4,np.random.default_rng(75),state)
    assert d['rejection']=='cheap_metropolis' and not calls and s is state;np.testing.assert_array_equal(y,x)


def test_selector_failure_propagates_and_invalid_endpoint_rejects(monkeypatch):
    x=np.array([-.4,.4,0.]);state=support.selected_state(x,0.,2.)
    bad=np.array([exact.inverse_branches(.05,2.)[-1],exact.inverse_branches(.05,2.)[-1],min(exact.inverse_branches(-.1,2.),key=abs)])
    monkeypatch.setattr(block,'score_block_kernel',lambda *a:(bad,1,1,1,0.))
    class Zero:
        def random(self):return 0.
    y,s,d=block.block_step(x,.1,0.,2.,.1,4,Zero(),state)
    assert d['rejection']=='global_selection' and d['selector_called'];np.testing.assert_array_equal(y,x)
    def fail(*a,**kw):raise StudentMLEError('selection_unresolved','injected')
    monkeypatch.setattr(block,'selected_state',fail)
    with pytest.raises(StudentMLEError):block.block_step(x,.1,0.,2.,.1,4,Zero(),state)
    with pytest.raises(FloatingPointError,match='budget exhausted'):block.score_pair_kernel(x.copy(),0,1,.1,0.,2.,.1,0)


@pytest.mark.parametrize('n,t,m',[(3,0.,1),(3,0.,16),(10,0.,4),(10,1.25,16)])
def test_chains_constraints_validity_fixed_seeds_and_independence(n,t,m):
    x=initial(n,t);p=dict(n=n,k=2.,num_iterations_T=60,initial_x=x,initial_mu=.2,prior_mean=0.,prior_std=2.,proposal_std_mu=.45,proposal_std_z=.1,block_length=m)
    before=np.random.get_state();a=block.run_gibbs(71,t,p,False);b=block.run_gibbs(72,t,p,False);c=block.run_gibbs(71,t,p,False)
    np.testing.assert_array_equal(a['mu_chain'],c['mu_chain']);np.testing.assert_array_equal(a['x_chain'],c['x_chain']);assert not np.array_equal(a['mu_chain'],b['mu_chain'])
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    for y in a['x_chain']:
        assert abs(np.sum((y-t)/(2+(y-t)**2)))<1e-12
        assert abs(global_mle(y,2.)-t)<1e-10
    assert a['counts']['selector_calls']<=61 and a['counts']['score_pairs']==60*m
