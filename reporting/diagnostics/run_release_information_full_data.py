"""Build paired full-data vs MLE-only posteriors for release-information analysis.

This script simulates observed datasets, computes p(mu | x_1:n) by one-dimensional
grid integration, and computes p(mu | hat_mu) by raw weighted Monte Carlo from
cached centered-MLE simulations. It writes the runset contract consumed by
``analyze_release_information.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import root_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kde_ref.moments import weighted_quantile
from models import loc_logistic, loc_student


TAIL_THRESHOLDS = [2.0, 3.0, 5.0, 10.0]
QUANTILES = [0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99]
Q_NAMES = ["q01", "q025", "q05", "q50", "q95", "q975", "q99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/release_information_runs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/release_information_runs/mle_error_cache"))
    parser.add_argument("--num-datasets", type=int, default=50)
    parser.add_argument("--mle-simulations", type=int, default=50000)
    parser.add_argument("--posterior-draws", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=2401)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-sd", type=float, default=10.0)
    parser.add_argument("--true-mu", type=float, default=0.0)
    parser.add_argument("--models", nargs="+", default=["normal_known_var", "logistic", "student_t", "laplace"])
    parser.add_argument("--n-values", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--student-k-values", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--laplace-n-values", nargs="+", type=int, default=[11, 21, 51])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def k_key(k: float | None) -> str:
    if k is None or not np.isfinite(k):
        return "NA"
    value = float(k)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def case_id(model: str, k: float | None, n: int, dataset_id: int) -> str:
    if model == "student_t":
        return f"{model}_k{k_key(k)}_n{n}_dataset{dataset_id:03d}"
    return f"{model}_n{n}_dataset{dataset_id:03d}"


def regimes(args: argparse.Namespace) -> list[tuple[str, float | None, int]]:
    out: list[tuple[str, float | None, int]] = []
    for model in args.models:
        if model == "student_t":
            for k in args.student_k_values:
                for n in args.n_values:
                    out.append((model, float(k), int(n)))
        elif model == "laplace":
            for n in args.laplace_n_values:
                out.append((model, None, int(n)))
        else:
            for n in args.n_values:
                out.append((model, None, int(n)))
    return out


def simulate_observed_data(rng: np.random.Generator, model: str, k: float | None, n: int, true_mu: float) -> np.ndarray:
    if model == "normal_known_var":
        return rng.normal(loc=true_mu, scale=1.0, size=n)
    if model == "student_t":
        return rng.standard_t(df=float(k), size=n) + true_mu
    if model == "logistic":
        return rng.logistic(loc=true_mu, scale=1.0, size=n)
    if model == "laplace":
        return rng.laplace(loc=true_mu, scale=1.0, size=n)
    raise ValueError(model)


def logistic_mle(x: np.ndarray) -> float:
    def score(mu: float) -> float:
        return float(np.sum(np.tanh((x - mu) / 2.0)))

    result = root_scalar(score, bracket=(float(np.min(x) - 10.0), float(np.max(x) + 10.0)), method="brentq")
    if not result.converged:
        raise RuntimeError("Logistic MLE root finding failed.")
    return float(result.root)


def compute_mle(x: np.ndarray, model: str, k: float | None) -> float:
    if model == "normal_known_var":
        return float(np.mean(x))
    if model == "student_t":
        return float(loc_student.get_mle(x, {"k": float(k), "n": int(len(x))}))
    if model == "logistic":
        return logistic_mle(x)
    if model == "laplace":
        return float(np.median(x))
    raise ValueError(model)


def centered_mle_errors(args: argparse.Namespace, model: str, k: float | None, n: int) -> np.ndarray:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = args.cache_dir / f"{model}_k{k_key(k)}_n{n}_B{args.mle_simulations}_seed{args.seed}.npz"
    if path.exists():
        return np.asarray(np.load(path)["z"], dtype=float)
    rng = np.random.default_rng(args.seed + 1009 * n + 9173 * (0 if k is None or not np.isfinite(k) else int(10 * float(k))))
    if model == "normal_known_var":
        z = rng.normal(loc=0.0, scale=1.0 / math.sqrt(n), size=args.mle_simulations)
    elif model == "laplace":
        data = rng.laplace(loc=0.0, scale=1.0, size=(args.mle_simulations, n))
        z = np.median(data, axis=1)
    elif model == "logistic":
        z = np.empty(args.mle_simulations, dtype=float)
        for i in range(args.mle_simulations):
            z[i] = loc_logistic.get_mle(rng.logistic(loc=0.0, scale=1.0, size=n), {"n": n})
    elif model == "student_t":
        z = np.empty(args.mle_simulations, dtype=float)
        params = {"k": float(k), "n": n}
        for i in range(args.mle_simulations):
            z[i] = loc_student.get_mle(rng.standard_t(df=float(k), size=n), params)
    else:
        raise ValueError(model)
    np.savez_compressed(path, z=z)
    return z


def log_likelihood_grid(model: str, k: float | None, x: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
    if model == "normal_known_var":
        return np.sum(stats.norm.logpdf(x[:, None], loc=mu_grid[None, :], scale=1.0), axis=0)
    if model == "student_t":
        return np.sum(stats.t.logpdf(x[:, None], df=float(k), loc=mu_grid[None, :], scale=1.0), axis=0)
    if model == "logistic":
        return np.sum(stats.logistic.logpdf(x[:, None], loc=mu_grid[None, :], scale=1.0), axis=0)
    if model == "laplace":
        return np.sum(stats.laplace.logpdf(x[:, None], loc=mu_grid[None, :], scale=1.0), axis=0)
    raise ValueError(model)


def cdf_from_density(mu_grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(mu_grid) / 2.0)])
    if cdf[-1] > 0 and np.isfinite(cdf[-1]):
        cdf = cdf / cdf[-1]
    return np.clip(cdf, 0.0, 1.0)


def grid_summary(mu_grid: np.ndarray, log_unnorm: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    log_unnorm = np.asarray(log_unnorm, dtype=float)
    log_max = float(np.nanmax(log_unnorm))
    weights = np.exp(log_unnorm - log_max)
    scaled_integral = float(np.trapezoid(weights, mu_grid))
    density = weights / scaled_integral
    cdf = cdf_from_density(mu_grid, density)
    mean = float(np.trapezoid(mu_grid * density, mu_grid))
    second = float(np.trapezoid(mu_grid * mu_grid * density, mu_grid))
    var = max(second - mean * mean, 0.0)
    qs = np.interp(QUANTILES, cdf, mu_grid)
    summary = {
        "mean": mean,
        "sd": float(np.sqrt(var)),
        "var": float(var),
        "normalization_constant": float(scaled_integral * np.exp(log_max)),
        "posterior_integral_check": float(np.trapezoid(density, mu_grid)),
    }
    summary.update(dict(zip(Q_NAMES, [float(q) for q in qs], strict=True)))
    return summary, density, cdf


def full_data_posterior(args: argparse.Namespace, x: np.ndarray, model: str, k: float | None, mu_hat: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if model == "normal_known_var":
        prior_var = args.prior_sd**2
        post_var = 1.0 / (1.0 / prior_var + len(x))
        post_mean = post_var * (args.prior_mean / prior_var + float(np.sum(x)))
        qs = stats.norm.ppf(QUANTILES, loc=post_mean, scale=np.sqrt(post_var))
        summary = {"mean": float(post_mean), "sd": float(np.sqrt(post_var)), "var": float(post_var)}
        summary.update(dict(zip(Q_NAMES, [float(q) for q in qs], strict=True)))
        draws = np.linspace(0.5 / args.posterior_draws, 1.0 - 0.5 / args.posterior_draws, args.posterior_draws)
        return summary, stats.norm.ppf(draws, loc=post_mean, scale=np.sqrt(post_var)), np.array([])

    lo = min(args.prior_mean - 6.0 * args.prior_sd, mu_hat - 20.0, float(np.median(x) - 20.0))
    hi = max(args.prior_mean + 6.0 * args.prior_sd, mu_hat + 20.0, float(np.median(x) + 20.0))
    for _ in range(3):
        mu_grid = np.linspace(lo, hi, args.grid_size)
        log_prior = stats.norm.logpdf(mu_grid, loc=args.prior_mean, scale=args.prior_sd)
        log_post = log_prior + log_likelihood_grid(model, k, x, mu_grid)
        peak = int(np.nanargmax(log_post))
        if peak < 5:
            lo -= hi - lo
        elif peak > args.grid_size - 6:
            hi += hi - lo
        else:
            break
    summary, _density, cdf = grid_summary(mu_grid, log_post)
    probs = np.linspace(0.5 / args.posterior_draws, 1.0 - 0.5 / args.posterior_draws, args.posterior_draws)
    draws = np.interp(probs, cdf, mu_grid)
    return summary, draws, mu_grid


def mle_only_posterior(args: argparse.Namespace, z: np.ndarray, model: str, k: float | None, n: int, mu_hat: float) -> tuple[dict[str, float], np.ndarray]:
    if model == "normal_known_var":
        prior_var = args.prior_sd**2
        post_var = 1.0 / (1.0 / prior_var + n)
        post_mean = post_var * (args.prior_mean / prior_var + n * mu_hat)
        qs = stats.norm.ppf(QUANTILES, loc=post_mean, scale=np.sqrt(post_var))
        summary = {"mean": float(post_mean), "sd": float(np.sqrt(post_var)), "var": float(post_var), "weighted_ess": np.nan}
        summary.update(dict(zip(Q_NAMES, [float(q) for q in qs], strict=True)))
        probs = np.linspace(0.5 / args.posterior_draws, 1.0 - 0.5 / args.posterior_draws, args.posterior_draws)
        return summary, stats.norm.ppf(probs, loc=post_mean, scale=np.sqrt(post_var))

    particles = float(mu_hat) - np.asarray(z, dtype=float)
    log_w = stats.norm.logpdf(particles, loc=args.prior_mean, scale=args.prior_sd)
    log_w -= float(np.nanmax(log_w))
    weights = np.exp(log_w)
    probs = weights / float(np.sum(weights))
    mean = float(np.sum(probs * particles))
    var = float(np.sum(probs * (particles - mean) ** 2))
    qs = weighted_quantile(particles, weights, QUANTILES)
    summary = {
        "mean": mean,
        "sd": float(np.sqrt(max(var, 0.0))),
        "var": float(max(var, 0.0)),
        "weighted_ess": float(np.sum(weights) ** 2 / np.sum(weights * weights)),
        "mle_error_B": int(z.size),
    }
    summary.update(dict(zip(Q_NAMES, [float(q) for q in qs], strict=True)))
    rng = np.random.default_rng(args.seed + int(abs(mu_hat) * 1_000_000) + 17 * n)
    draw_idx = rng.choice(np.arange(particles.size), size=args.posterior_draws, replace=True, p=probs)
    return summary, particles[draw_idx]


def predictive_tail(model: str, k: float | None, mu_values: np.ndarray, center: float, threshold: float) -> float:
    if model == "normal_known_var":
        probs = stats.norm.cdf(center - threshold, loc=mu_values, scale=1.0) + stats.norm.sf(center + threshold, loc=mu_values, scale=1.0)
    elif model == "student_t":
        probs = stats.t.cdf(center - threshold, df=float(k), loc=mu_values, scale=1.0) + stats.t.sf(center + threshold, df=float(k), loc=mu_values, scale=1.0)
    elif model == "logistic":
        probs = stats.logistic.cdf(center - threshold, loc=mu_values, scale=1.0) + stats.logistic.sf(center + threshold, loc=mu_values, scale=1.0)
    elif model == "laplace":
        probs = stats.laplace.cdf(center - threshold, loc=mu_values, scale=1.0) + stats.laplace.sf(center + threshold, loc=mu_values, scale=1.0)
    else:
        return np.nan
    return float(np.mean(probs))


def summary_row(base: dict[str, Any], conditioning: str, method: str, summary: dict[str, float], draws: np.ndarray, mu_hat: float) -> dict[str, Any]:
    row = {**base, "conditioning": conditioning, "method": method, "draws": int(draws.size), "mu_star": float(mu_hat), **summary}
    for threshold in TAIL_THRESHOLDS:
        row[f"predictive_tail_prob_gt_{threshold:g}"] = predictive_tail(base["model"], base.get("k"), draws, mu_hat, threshold)
    return row


def chain_frame(base: dict[str, Any], conditioning: str, method: str, draws: np.ndarray, mu_hat: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{key: value for key, value in base.items() if key not in {"source_file"}},
            "conditioning": conditioning,
            "method": method,
            "iteration": np.arange(draws.size, dtype=int),
            "mu": draws,
            "is_burn_in": False,
            "mu_star": float(mu_hat),
        }
    )


def observed_frame(base: dict[str, Any], x: np.ndarray, mu_hat: float, true_mu: float) -> pd.DataFrame:
    row = {**base, "conditioning": "observed_data", "mu_star": float(mu_hat), "true_mu": float(true_mu)}
    for i, value in enumerate(x):
        row[f"x_{i}"] = float(value)
    abs_dev = np.abs(x - mu_hat)
    row["actual_max_abs"] = float(np.max(abs_dev))
    for threshold in TAIL_THRESHOLDS:
        row[f"actual_count_gt_{threshold:g}"] = int(np.sum(abs_dev > threshold))
    return pd.DataFrame([row])


def write_case(args: argparse.Namespace, base: dict[str, Any], x: np.ndarray, mu_hat: float, full: tuple, mle: tuple) -> None:
    cid = str(base["case_id"])
    out = args.out_dir / f"case_{cid}"
    if out.exists() and not args.force:
        return
    out.mkdir(parents=True, exist_ok=True)
    full_summary, full_draws, _ = full
    mle_summary, mle_draws = mle
    full_method = "grid_full_data" if base["model"] != "normal_known_var" else "exact_normal_full_data"
    mle_method = "raw_weighted_mle_mc" if base["model"] != "normal_known_var" else "exact_normal_mle_only"
    pd.DataFrame([summary_row(base, "full_data", full_method, full_summary, full_draws, mu_hat)]).to_csv(out / "full_data_posterior_summaries.csv", index=False)
    pd.DataFrame([summary_row(base, "mle_only", mle_method, mle_summary, mle_draws, mu_hat)]).to_csv(out / "mle_only_posterior_summaries.csv", index=False)
    chain_frame(base, "full_data", full_method, full_draws, mu_hat).to_csv(out / "full_data_chain_samples.csv", index=False)
    chain_frame(base, "mle_only", mle_method, mle_draws, mu_hat).to_csv(out / "mle_only_chain_samples.csv", index=False)
    observed_frame(base, x, mu_hat, args.true_mu).to_csv(out / "observed_data.csv", index=False)
    metadata = {
        **base,
        "status": "completed",
        "true_mu": float(args.true_mu),
        "prior_mean": float(args.prior_mean),
        "prior_sd": float(args.prior_sd),
        "mle_simulations": int(args.mle_simulations),
        "posterior_draws": int(args.posterior_draws),
        "grid_size": int(args.grid_size),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for model, k, n in regimes(args):
        print(f"regime model={model} k={k_key(k)} n={n}: preparing centered MLE errors")
        z = centered_mle_errors(args, model, k, n)
        for dataset_idx in range(args.num_datasets):
            x = simulate_observed_data(rng, model, k, n, args.true_mu)
            mu_hat = compute_mle(x, model, k)
            cid = case_id(model, k, n, dataset_idx)
            base = {
                "case_id": cid,
                "dataset_id": cid,
                "model": model,
                "k": np.nan if k is None else float(k),
                "k_key": k_key(k),
                "n": int(n),
                "seed": int(args.seed),
                "dataset_seed": int(dataset_idx),
                "initialization": "not_applicable",
            }
            full = full_data_posterior(args, x, model, k, mu_hat)
            mle = mle_only_posterior(args, z, model, k, n, mu_hat)
            write_case(args, base, x, mu_hat, full, mle)
            rows.append(
                {
                    **base,
                    "mu_hat": float(mu_hat),
                    "true_mu": float(args.true_mu),
                    "actual_max_abs": float(np.max(np.abs(x - mu_hat))),
                    "full_mean": float(full[0]["mean"]),
                    "mle_only_mean": float(mle[0]["mean"]),
                    "full_sd": float(full[0]["sd"]),
                    "mle_only_sd": float(mle[0]["sd"]),
                }
            )
        print(f"completed model={model} k={k_key(k)} n={n}")
    manifest = {
        "num_cases": len(rows),
        "num_datasets_per_regime": int(args.num_datasets),
        "mle_simulations": int(args.mle_simulations),
        "posterior_draws": int(args.posterior_draws),
        "regimes": [{"model": m, "k": None if k is None else float(k), "n": n} for m, k, n in regimes(args)],
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "release_information_run_manifest.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
