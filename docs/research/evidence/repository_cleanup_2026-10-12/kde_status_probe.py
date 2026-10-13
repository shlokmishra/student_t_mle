"""Deterministic independent mixture-CDF check; prints, never rewrites evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
from scipy import optimize, stats
from kde_ref.moments import gaussian_kde_gaussian_prior_moments
z = np.array([-1., 1.])
kde = stats.gaussian_kde(z, bw_method='scott')
h = float(np.sqrt(kde.covariance[0, 0]))
reported = gaussian_kde_gaussian_prior_moments(SimpleNamespace(samples=z, bandwidth=h), 0., 0., 2.)
variance = 1. / (1. / 4. + 1. / h**2)
means = variance * (-z / h**2)
normalizer_components = stats.norm.pdf(-z, scale=np.sqrt(4. + h**2))
weights = normalizer_components / normalizer_components.sum()
def cdf(value):
    return float(np.sum(weights * stats.norm.cdf(value, loc=means, scale=np.sqrt(variance))))
quantiles = [optimize.brentq(lambda v: cdf(v)-p, -20., 20.) for p in [.025, .5, .975]]
result = dict(helper_reported=reported, correct_mixture_quantiles=quantiles, correct_normalizer=float(normalizer_components.mean()))
expected = json.loads(Path(__file__).with_name('kde_status.json').read_text())
np.testing.assert_allclose(quantiles, expected['correct_mixture_quantiles'], atol=1e-12)
np.testing.assert_allclose(result['correct_normalizer'], expected['correct_normalizer'], atol=1e-12)
print(json.dumps(result, indent=2))
