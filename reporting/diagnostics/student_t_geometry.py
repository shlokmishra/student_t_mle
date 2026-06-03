"""Shared Student-t location MLE geometry helpers for diagnostic scripts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.stats as stats

EPS = 1e-12


def psi(y: np.ndarray | float, k: float) -> np.ndarray | float:
    y_arr = np.asarray(y, dtype=float)
    out = y_arr / (k + y_arr * y_arr)
    return float(out) if np.isscalar(y) else out


def psi_prime(y: np.ndarray | float, k: float) -> np.ndarray | float:
    y_arr = np.asarray(y, dtype=float)
    denom = k + y_arr * y_arr
    out = (k - y_arr * y_arr) / (denom * denom)
    return float(out) if np.isscalar(y) else out


def constraint_value(x: np.ndarray, mu_star: float, k: float) -> float:
    return float(np.sum(psi(np.asarray(x, dtype=float) - mu_star, k)))


def grad_constraint(x: np.ndarray, mu_star: float, k: float) -> np.ndarray:
    return np.asarray(psi_prime(np.asarray(x, dtype=float) - mu_star, k), dtype=float)


def gram(x: np.ndarray, mu_star: float, k: float) -> float:
    g = grad_constraint(x, mu_star, k)
    return float(np.dot(g, g))


def loglik_x_given_mu(x: np.ndarray, mu: float, k: float) -> float:
    return float(np.sum(stats.t.logpdf(np.asarray(x, dtype=float), df=k, loc=mu, scale=1.0)))


def potential_without_gram(x: np.ndarray, mu: float, k: float) -> float:
    y = np.asarray(x, dtype=float) - mu
    return float(0.5 * (k + 1.0) * np.sum(np.log1p((y * y) / k)))


def potential_with_gram(x: np.ndarray, mu: float, mu_star: float, k: float) -> float:
    return float(potential_without_gram(x, mu, k) + 0.5 * np.log(max(gram(x, mu_star, k), 1e-300)))


def geometry_summary(x: np.ndarray, mu_star: float, k: float) -> dict[str, Any]:
    y = np.asarray(x, dtype=float) - float(mu_star)
    gp = grad_constraint(x, mu_star, k)
    G = gram(x, mu_star, k)
    ps = np.asarray(psi(y, k), dtype=float)
    return {
        "constraint_residual": abs(constraint_value(x, mu_star, k)),
        "gram": G,
        "log_gram": float(np.log(max(G, 1e-300))),
        "max_abs_y": float(np.max(np.abs(y))),
        "count_abs_y_gt_sqrt_k": int(np.sum(np.abs(y) > np.sqrt(k))),
        "count_abs_y_gt_5": int(np.sum(np.abs(y) > 5.0)),
        "min_abs_psi_prime": float(np.min(np.abs(gp))),
        "psi_mean": float(np.mean(ps)),
        "psi_sd": float(np.std(ps)),
        "psi_min": float(np.min(ps)),
        "psi_max": float(np.max(ps)),
    }


def posterior_summary(samples: Iterable[float], prefix: str = "") -> dict[str, float]:
    arr = np.asarray(list(samples), dtype=float)
    keys = ["mean", "var", "sd", "q025", "q50", "q975"]
    if arr.size == 0:
        return {prefix + key: float("nan") for key in keys}
    out = {
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q50": float(np.quantile(arr, 0.5)),
        "q975": float(np.quantile(arr, 0.975)),
    }
    return {prefix + key: val for key, val in out.items()}


def tail_mass_from_pdf(pdf, grid: np.ndarray, center: float, radius: float) -> float:
    vals = np.maximum(np.asarray(pdf(grid), dtype=float), 0.0)
    vals = vals / max(float(np.trapezoid(vals, grid)), EPS)
    mask = np.abs(grid - center) > radius
    return float(np.trapezoid(vals[mask], grid[mask])) if np.any(mask) else 0.0


def kde_pdf_summary(pdf, grid: np.ndarray, prefix: str = "") -> dict[str, float]:
    vals = np.maximum(np.asarray(pdf(grid), dtype=float), 0.0)
    vals = vals / max(float(np.trapezoid(vals, grid)), EPS)
    cdf = np.cumsum((vals[:-1] + vals[1:]) * np.diff(grid) / 2.0)
    cdf = np.concatenate([[0.0], cdf])
    cdf = cdf / max(cdf[-1], EPS)
    mean = float(np.trapezoid(grid * vals, grid))
    var = float(np.trapezoid((grid - mean) ** 2 * vals, grid))
    out = {
        "mean": mean,
        "var": var,
        "sd": float(np.sqrt(max(var, 0.0))),
        "q025": float(np.interp(0.025, cdf, grid)),
        "q50": float(np.interp(0.5, cdf, grid)),
        "q975": float(np.interp(0.975, cdf, grid)),
    }
    return {prefix + key: val for key, val in out.items()}
