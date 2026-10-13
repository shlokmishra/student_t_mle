"""Protect benchmark accounting and unresolved-evidence handling."""
import numpy as np
from scripts.compare_location_only_performance import (
    bank, binomial_interval, weighted_function_ess, passes_screens, parameters,
)


def test_iid_equivalent_ess_matches_snis_variance():
    x = np.array([-1., .2, 2., 5.])
    w = np.array([.2, .5, 1., .1])
    a = w/w.sum()
    expected = (a@((x-a@x)**2))/np.sum(a*a*(x-a@x)**2)
    assert abs(weighted_function_ess(x, w)-expected)<1e-14
    assert abs(weighted_function_ess(x, np.ones(4))-4)<1e-14
    assert weighted_function_ess(np.zeros(4), w) is None


def test_zero_events_have_nonzero_interval():
    assert binomial_interval(0, 100)[1] > .03
    assert binomial_interval(100, 100)[0] < .97
    assert binomial_interval(0, 0) == [None, None]


def test_screen_does_not_pass_missing_or_unresolved_metrics():
    cfg = dict(required_metrics=["mean","latent_tail6"], difference_mcse_max=4.,
               rhat_max=1.01, ess_min=400, minimum_tail_events=20)
    row = dict(difference_mcse=dict(mean=0.,latent_tail6=.1),
               diagnostics=dict(mu=dict(rhat=1.,ess_bulk=500,ess_tail=500)),
               tail6_events_per_chain=[10,10,10,10])
    assert passes_screens(row,cfg)["all_pass"]
    row["difference_mcse"]["latent_tail6"] = None
    assert not passes_screens(row,cfg)["all_pass"]
    row["difference_mcse"]["latent_tail6"] = 0.
    row["tail6_events_per_chain"] = [0,10,10,10]
    assert not passes_screens(row,cfg)["all_pass"]


def test_small_independent_banks_and_frozen_parameters():
    for family in ["student","logistic","laplace"]:
        x,t,meta = bank(family,3,8,723,10)
        y,u,_ = bank(family,3,8,723,10)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(t,u)
        if family=="laplace":
            np.testing.assert_array_equal(t,np.median(x,axis=1))
        assert meta["estimator_calls"]==8 and meta["failures"]==0
    case=dict(n=3,family="logistic",prior_mean=0.,prior_std=2.)
    settings=dict(proposal_std_mu=.45,proposal_std_z=.1,student_logistic_block_n3=4,
                  student_logistic_block_other=16,student_rattle_step_size=.1,
                  logistic_laplace_step_size=.05,rattle_num_steps=10)
    p=parameters(case,dict(settings=settings,chain_ceiling_seconds=120),
                 np.array([-.3,0.,.3]),0.,"block",12)
    assert p["block_length"]==4 and p["rattle_step_size"]==.05
    assert p["proposal_std_mu"]==.45 and p["num_iterations_T"]==12
