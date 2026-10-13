"""Independent chart/flux calculations and selected-MLE kernel regressions."""
import itertools
import math

import numpy as np
import pytest
from scipy import stats

from models.loc.student import selected_gibbs as selected
from models.loc.student.mle import global_mle, StudentMLEError


def data_from_scores(scores, branches=None, k=2., t=0.):
    branches = branches or [0]*len(scores)
    return np.array([t+selected.inverse_branches(z, k)[b] for z, b in zip(scores, branches)])


def inner_data(scores, k=2., t=0.):
    return np.array([t+min(selected.inverse_branches(z, k), key=abs) for z in scores])


def params(x, iterations=100, **kwargs):
    return dict(n=len(x), k=2., num_iterations_T=iterations, initial_x=x,
                initial_mu=.2, prior_mean=0., prior_std=2.,
                proposal_std_mu=.4, proposal_std_z=.1, **kwargs)


def independent_branch_log(y, mu, t, k=2.):
    derivative = (k-y*y)/(k+y*y)**2
    return float(stats.t.logpdf(t+y-mu, df=k)-math.log(abs(derivative)))


@pytest.mark.parametrize('k,z', [(1., .01), (2., .01), (3., -.1), (2., 1e-150), (2., 0.)])
def test_inverse_branches_are_finite_and_recover_score(k, z):
    y = selected.inverse_branches(z, k)
    assert len(y) == (1 if z == 0 else 2)
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y/(k+y*y), z, rtol=4e-15, atol=0)


def test_cauchy_outer_probability_is_half_and_zero_is_singleton():
    for z in [.01, -.1, .4]:
        y, p, q = selected.branch_distribution(z, 0., 0., 1.)
        np.testing.assert_allclose(p, [.5, .5], atol=3e-15)
    y, p, q = selected.branch_distribution(0., 0., 0., 2.)
    np.testing.assert_array_equal(y, [0.]); np.testing.assert_array_equal(p, [1.])
    assert q == math.log(2.)


@pytest.mark.parametrize('z,mu,t', [(.02, .7, 0.), (-.2, -.3, 1.25), (.31, -2., 0.)])
def test_branch_probabilities_match_independent_jacobian_weights(z, mu, t):
    y, p, q = selected.branch_distribution(z, mu, t, 2.)
    logs = np.array([independent_branch_log(v, mu, t) for v in y])
    expected = np.exp(logs-logs.max()); expected /= expected.sum()
    np.testing.assert_allclose(p, expected, rtol=3e-14)
    uniforms = np.random.default_rng(71).random(20000)
    empirical = np.mean(uniforms < p[0])
    assert abs(empirical-p[0]) < 6*math.sqrt(p[0]*(1-p[0])/len(uniforms))+.001


def test_joint_ratio_equals_full_target_and_proposal_flux_all_branches():
    x = inner_data([.12, -.02, -.1])
    info = selected.target_info(x, 0., 2.)
    zi, delta, znew, sigma, mu = .12, .1, .05, .11, .35
    low, high = selected.feasible_interval(delta, 2.)
    def scalar_h(a, b):
        return stats.truncnorm.logpdf(b, (low-a)/sigma, (high-a)/sigma, loc=a, scale=sigma)
    def branch_log_probability(z, y):
        ys = selected.inverse_branches(z, 2.)
        logs = np.array([independent_branch_log(v, mu, 0.) for v in ys])
        q = np.logaddexp.reduce(logs)
        return independent_branch_log(y, mu, 0.)-q
    def chart_target(state, ti):
        return sum(independent_branch_log(state[j], mu, 0.) for j in [0, 1])+ti.log_r
    positive = invalid = 0
    for bi, bj in itertools.product(range(2), repeat=2):
        proposed = x.copy()
        proposed[0] = selected.inverse_branches(znew, 2.)[bi]
        proposed[1] = selected.inverse_branches(delta-znew, 2.)[bj]
        pi = selected.target_info(proposed, 0., 2.)
        ratio, parts = selected.pair_log_acceptance(x, proposed, (0, 1), mu, 0., 2., sigma, info, pi)
        if not pi.selected:
            assert ratio == -math.inf; invalid += 1; continue
        positive += 1
        forward = scalar_h(zi, znew)+branch_log_probability(znew, proposed[0])+branch_log_probability(delta-znew, proposed[1])
        reverse = scalar_h(znew, zi)+branch_log_probability(zi, x[0])+branch_log_probability(delta-zi, x[1])
        full_ratio = chart_target(proposed, pi)-chart_target(x, info)+reverse-forward
        np.testing.assert_allclose(ratio, full_ratio, atol=3e-12)
        back, _ = selected.pair_log_acceptance(proposed, x, (0, 1), mu, 0., 2., sigma, pi, info)
        np.testing.assert_allclose(back, -ratio, atol=3e-12)
        left = chart_target(x, info)+forward+min(0., ratio)
        right = chart_target(proposed, pi)+reverse+min(0., -ratio)
        np.testing.assert_allclose(left, right, atol=3e-12)
    assert positive and invalid


def test_production_pair_step_uses_inspected_log_ratio(monkeypatch):
    x = inner_data([.12, -.02, -.1])
    info = selected.target_info(x, 0., 2.)
    rng = np.random.default_rng(731)
    captured = []
    original = selected.target_info
    def record(*args, **kwargs):
        ti = original(*args, **kwargs); captured.append((np.array(args[0]), ti)); return ti
    monkeypatch.setattr(selected, 'target_info', record)
    for _ in range(20):
        _, _, diag = selected.pair_step(x, (0, 1), .35, 0., 2., .11, rng, current_info=info)
        proposed, pi = captured[-1]
        ratio, _ = selected.pair_log_acceptance(x, proposed, (0, 1), .35, 0., 2., .11, info, pi)
        if math.isfinite(ratio):
            np.testing.assert_allclose(diag['log_acceptance_ratio'], ratio, atol=4e-12)
        else:
            assert diag['log_acceptance_ratio'] == ratio


def test_global_selection_is_required_even_for_local_maximum():
    x = np.array([4., 4., (-17+math.sqrt(33))/16])
    ti = selected.target_info(x, 0., 1.)
    assert ti.score_residual < 1e-15 and ti.curvature > 0
    assert not ti.selected and ti.log_r == -math.inf


def test_rejection_retains_both_branches_and_whole_dataset(monkeypatch):
    x = inner_data([.12, -.02, -.1])
    monkeypatch.setattr(selected, 'propose_score', lambda z, sigma, low, high, rng: (.05, selected.truncnorm_log_mass(z, sigma, low, high)))
    class Fixed:
        def __init__(self): self.values = iter([.999999, .999999, .001])
        def random(self): return next(self.values)
    new, ti, diag = selected.pair_step(x, (0, 1), 0., 0., 2., .1, Fixed())
    assert not diag['selected_proposal'] and not diag['accepted']
    np.testing.assert_array_equal(new, x)


def test_selector_failures_propagate_without_becoming_rejections(monkeypatch):
    x = inner_data([.12, -.02, -.1]); ti = selected.target_info(x, 0., 2.)
    def fail(*args, **kwargs): raise StudentMLEError('selection_unresolved', 'injected failure')
    monkeypatch.setattr(selected, 'global_mle', fail)
    with pytest.raises(StudentMLEError, match='selection_unresolved'):
        selected.pair_step(x, (0, 1), .2, 0., 2., .1, np.random.default_rng(73), current_info=ti)


def test_stale_correction_cache_cannot_be_used():
    x = inner_data([.12, -.02, -.1]); ti = selected.target_info(x, 0., 2.)
    new = inner_data([.2, -.1, -.1])
    with pytest.raises(ValueError, match='cached correction'):
        selected.pair_step(new, (0, 1), .2, 0., 2., .1, np.random.default_rng(73), current_info=ti)


def test_no_fold_strip_and_degenerate_curvature_is_explicit():
    bound = 1/(2*math.sqrt(2.))
    near = bound-1e-14  # inside legacy's excluded 1e-12 strip
    y, p, q = selected.branch_distribution(near, 0., 0., 2.)
    assert len(y) == 2 and np.isfinite(q) and np.isfinite(p).all()
    with pytest.raises(selected.SelectedMLENumericalError, match='score_chart_boundary'):
        selected.inverse_branches(bound, 2.)
    with pytest.raises(selected.SelectedMLENumericalError, match='curvature_degenerate'):
        selected.target_info([-1., -1., 1., 1.], 0., 1.)
    near_flat = selected.target_info(np.array([-1., -1., 1., 1.])*(1-1e-12), 0., 1.)
    assert 0 < near_flat.curvature < 1e-10 and np.isfinite(near_flat.log_r)


@pytest.mark.parametrize('bad', [[], [0, 1], [0, np.nan, 1], [0, 1j, 1]])
def test_invalid_data_explicit(bad):
    with pytest.raises(ValueError): selected.target_info(bad, 0., 2.)


def test_score_offset_is_not_silently_normalized():
    with pytest.raises(selected.SelectedMLENumericalError, match='score_constraint'):
        selected.target_info([.1, .2, -.1], 0., 2.)


@pytest.mark.parametrize('n,t,scale', [(3, 0., 1.), (10, 0., 1.), (10, 1.25, 1.), (3, .75, 2.)])
def test_chain_constraint_selected_mle_and_reproducibility(n, t, scale):
    rng = np.random.default_rng(99+n)
    e = rng.standard_t(2., n); x = t+scale*(e-global_mle(e, 2.))
    p = params(x, iterations=120, known_scale=scale)
    a = selected.run_gibbs(744, t, p, verbose=False)
    # An intervening independent chain must not mutate chain 744's RNG stream.
    b = selected.run_gibbs(745, t, p, verbose=False)
    c = selected.run_gibbs(744, t, p, verbose=False)
    np.testing.assert_array_equal(a['mu_chain'], c['mu_chain'])
    np.testing.assert_array_equal(a['x_chain'], c['x_chain'])
    assert not np.array_equal(a['mu_chain'], b['mu_chain'])
    assert a['target'] == 'selected_global_student_location_mle'
    for state in a['x_chain']:
        assert abs(selected.score_coordinates(state, t, 2., scale).sum()) < 2e-12
        assert abs(scale*global_mle(state/scale, 2.)-t) < 2e-11
    assert a['min_retained_curvature'] > 0


def test_no_global_numpy_rng_side_effect_and_without_storing_latents():
    x = inner_data([.12, -.02, -.1])
    np.random.seed(73); before = np.random.get_state()
    a = selected.run_gibbs(746, 0., params(x, iterations=20, store_x_chain=False), verbose=False)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert 'x_chain' not in a and a['x_final'].shape == (3,)


def test_disjoint_pairs_use_the_updated_full_dataset_correction(monkeypatch):
    x = inner_data([.12, -.02, -.1, 0.])
    original = selected.pair_step
    seen = []
    def record(state, *args, **kwargs):
        assert kwargs['current_info'].state_key[0] == state.tobytes()
        result = original(state, *args, **kwargs)
        seen.append((state.copy(), result[0].copy(), result[2]['accepted']))
        return result
    monkeypatch.setattr(selected, 'pair_step', record)
    selected.run_gibbs(919, 0., params(x, iterations=20), False)
    assert any(accepted for _, _, accepted in seen)
    for left, right in zip(seen, seen[1:]):
        np.testing.assert_array_equal(left[1], right[0])


def test_parameter_density_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(selected, 'unnormalized_mu_logpdf', lambda *args: math.nan)
    with pytest.raises(selected.SelectedMLENumericalError, match='parameter_density_nonfinite'):
        selected.run_gibbs(811, 0., params(inner_data([.12, -.02, -.1]), iterations=1), False)
