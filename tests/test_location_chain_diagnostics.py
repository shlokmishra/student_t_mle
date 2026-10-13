"""Checks for full-chain diagnostics, quantile axes, and fixed constraints."""
import numpy as np
import pytest
from scripts.analyze_location_chains import observables, constraint_fixed, scalar_diagnostics, acf


def test_latent_quantiles_are_across_observations_not_iterations():
    x = np.array([[[1.,3.,5.], [3.,5.,11.]]])
    a = observables(np.zeros((1,2)), x, dict(n=3,release=1.))
    np.testing.assert_allclose(a["latent_q05"], [[.2,.2*1+2.]])
    np.testing.assert_allclose(a["latent_q50"], [[2.,4.]])
    np.testing.assert_allclose(a["latent_q95"], [[3.8,9.4]])
    np.testing.assert_array_equal(a["maximum"], [[4.,10.]])
    np.testing.assert_array_equal(a["tail6"], [[0.,1.]])
    b = observables(np.zeros((1,2)), x[...,::-1], dict(n=3,release=1.))
    np.testing.assert_array_equal(a["order1"], b["order1"])
    assert not np.array_equal(a["x1"], b["x1"])


def test_constraint_fixed_is_distinguished_from_stuck_chains():
    case = dict(family="laplace",n=3)
    assert constraint_fixed("latent_q50",case) and constraint_fixed("order2",case)
    assert not constraint_fixed("x2",case)
    assert constraint_fixed("latent_q50",dict(family="laplace",n=4))
    assert not constraint_fixed("latent_q50",dict(family="student",n=3))
    fixed = scalar_diagnostics(np.full((4,100),1e-15),fixed=True)
    assert fixed["status"]=="constraint_fixed" and fixed["rhat"] is None
    with pytest.raises(ArithmeticError):
        scalar_diagnostics(np.full((4,100),1e-4),fixed=True)
    assert scalar_diagnostics(np.zeros((4,100)))["status"]=="unresolved_constant"
    assert scalar_diagnostics(np.tile(np.arange(4.)[:,None],(1,100)))["status"].startswith("unresolved")


def test_diagnostics_expose_chain_separation_and_serial_dependence():
    rng = np.random.default_rng(1927)
    iid = rng.normal(size=(4,3000))
    good = scalar_diagnostics(iid)
    bad = scalar_diagnostics(iid+np.arange(4.)[:,None]*2)
    assert good["rhat"]<1.01 and good["ess_bulk"]>5000
    assert bad["rhat"]>1.3 and bad["ess_bulk"]<good["ess_bulk"]
    assert good["quantile_ess"]["q01"] is not None
    sticky = iid.copy()
    for i in range(1,sticky.shape[1]):
        sticky[:,i] += .98*sticky[:,i-1]
    assert scalar_diagnostics(sticky)["ess_tail"]<good["ess_tail"]
    assert acf(sticky)[:,1].mean()>.9
    assert abs(acf(iid)[:,1].mean())<.05
    np.testing.assert_allclose(acf(iid)[:,0],1.)


def test_nonfinite_data_is_not_silently_discarded():
    with pytest.raises(ValueError):
        scalar_diagnostics(np.full((4,100),np.nan))
