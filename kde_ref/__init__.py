# KDE reference posterior: p(theta | hat_theta*) from benchmark MLE samples + prior.

from .posterior import get_normalized_posterior_pdf, validate_posterior_1d

__all__ = ["get_normalized_posterior_pdf", "validate_posterior_1d"]
