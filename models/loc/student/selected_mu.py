"""Unfloored parameter MH for selected-MLE kernels; legacy helper stays frozen."""
from numba import njit
from .mu_mh import unnormalized_mu_logpdf
from models.shared.selected_numerics import mh_accept


@njit(cache=True)
def update_mu_mh_from_variates(mu, x, proposal_std, prior_mean, prior_std, k, noise, uniform):
    proposed = mu + proposal_std * noise
    ratio = (unnormalized_mu_logpdf(proposed, x, prior_mean, prior_std, k)
             - unnormalized_mu_logpdf(mu, x, prior_mean, prior_std, k))
    accept = mh_accept(ratio, uniform)
    return (proposed if accept else mu), int(accept)
