"""Audit KDE backend and bandwidth choices for final shared plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np
import scipy.stats as stats

from analysis import posterior_variance_from_kde
from kde_ref.posterior import (
    build_likelihood_kde_backend,
    get_normalized_posterior_pdf,
    validate_posterior_1d,
)
from models import loc_laplace, loc_logistic, loc_student
from validation import run_single_comparison


OUT_DIR = Path("artifacts/final_comparison")
OUT_JSON = OUT_DIR / "kde_bandwidth_audit.json"

MODEL_SPECS = [
    {
        "key": "loc_logistic",
        "label": "Logistic",
        "seed": 101,
        "tail_sensitive": False,
        "model": "loc_logistic",
        "module": loc_logistic,
        "n": 20,
        "model_kw": {},
        "plot_run": {
            "T_gibbs": 3000,
            "T_baseline": 3000,
            "T_kde": 10000,
            "T_fulldata": 3000,
            "burnin": 500,
            "base_params": {
                "rattle_step_size": 0.05,
                "rattle_num_steps": 2,
                "rattle_reverse_position_tol": 5e-3,
                "rattle_reverse_momentum_tol": 5e-3,
                "rattle_proj_damping": 1.0,
                "rattle_proj_line_search": True,
                "rattle_proj_init_strategy": "trial",
                "rattle_relaxed_position_tol": 5e-2,
                "rattle_relaxed_momentum_tol": 5e-2,
            },
        },
    },
    {
        "key": "loc_laplace",
        "label": "Laplace",
        "seed": 102,
        "tail_sensitive": False,
        "model": "loc_laplace",
        "module": loc_laplace,
        "n": 20,
        "model_kw": {"b": 1.0},
        "plot_run": None,
    },
    {
        "key": "loc_student_k3",
        "label": "Student-3",
        "seed": 103,
        "tail_sensitive": True,
        "model": "loc_student",
        "module": loc_student,
        "n": 20,
        "model_kw": {"k": 3.0},
        "plot_run": {
            "T_gibbs": 3000,
            "T_baseline": 3000,
            "T_kde": 10000,
            "T_fulldata": 3000,
            "burnin": 500,
            "base_params": {
                "rattle_step_size": 0.05,
                "rattle_num_steps": 1,
                "rattle_reverse_position_tol": 1e-2,
                "rattle_reverse_momentum_tol": 1e-2,
                "rattle_proj_damping": 1.0,
                "rattle_proj_line_search": True,
                "rattle_proj_init_strategy": "trial",
                "rattle_relaxed_position_tol": 5e-2,
                "rattle_relaxed_momentum_tol": 5e-2,
            },
        },
    },
    {
        "key": "loc_student_k2",
        "label": "Student-2",
        "seed": 104,
        "tail_sensitive": True,
        "model": "loc_student",
        "module": loc_student,
        "n": 20,
        "model_kw": {"k": 2.0},
        "plot_run": {
            "T_gibbs": 3000,
            "T_baseline": 3000,
            "T_kde": 10000,
            "T_fulldata": 3000,
            "burnin": 500,
            "base_params": {
                "rattle_step_size": 0.04,
                "rattle_num_steps": 2,
                "rattle_reverse_position_tol": 2e-2,
                "rattle_reverse_momentum_tol": 2e-2,
                "rattle_proj_damping": 1.0,
                "rattle_proj_line_search": True,
                "rattle_proj_init_strategy": "trial",
                "rattle_relaxed_position_tol": 5e-2,
                "rattle_relaxed_momentum_tol": 5e-2,
            },
        },
    },
    {
        "key": "loc_cauchy",
        "label": "Cauchy",
        "seed": 105,
        "tail_sensitive": True,
        "model": "loc_student",
        "module": loc_student,
        "n": 20,
        "model_kw": {"k": 1.0},
        "plot_run": None,
    },
]

CANDIDATES = [
    {"name": "scott", "bw_method": "scott", "complexity_rank": 0},
    {"name": "silverman", "bw_method": "silverman", "complexity_rank": 1},
    {"name": "SJ_transform", "bw_method": "SJ_transform", "complexity_rank": 2},
    {"name": "t_abram", "bw_method": "t_abram", "complexity_rank": 3},
]


def _default_params(n: int, model_kw: dict[str, Any]) -> dict[str, Any]:
    params = {
        "mu_true": 2.0,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "proposal_std_mu": 0.9,
        "proposal_std_z": 0.03,
        "kde_bw_method": "scott",
        "n": n,
        "num_iterations_T": 3000,
    }
    params.update(model_kw)
    return params


def _split_samples(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(0.7 * samples.size)
    return samples[:n_train], samples[n_train:]


def _score_candidate(train: np.ndarray, test: np.ndarray, params: dict[str, Any], bw_method: Any) -> dict[str, Any]:
    candidate_params = dict(params)
    candidate_params["kde_bw_method"] = bw_method
    backend = build_likelihood_kde_backend(train, candidate_params, verbose=False)
    test_logpdf = np.asarray(backend.logpdf(test), dtype=float)
    tail_mask = np.abs(test) >= np.quantile(np.abs(test), 0.9)
    tail_logpdf = test_logpdf[tail_mask]
    return {
        "heldout_mean_logpdf": float(np.mean(test_logpdf)),
        "heldout_se_logpdf": float(np.std(test_logpdf, ddof=1) / np.sqrt(test_logpdf.size)),
        "tail_mean_logpdf": float(np.mean(tail_logpdf)),
        "tail_count": int(tail_logpdf.size),
    }


def _posterior_summary(
    observed_mle: float,
    benchmark_samples: np.ndarray,
    params: dict[str, Any],
    bw_method: Any,
) -> dict[str, Any]:
    candidate_params = dict(params)
    candidate_params["kde_bw_method"] = bw_method
    posterior_pdf = get_normalized_posterior_pdf(observed_mle, candidate_params, benchmark_samples, verbose=False)
    mean, var = posterior_variance_from_kde(posterior_pdf)
    integral = validate_posterior_1d(posterior_pdf)
    return {
        "posterior_mean": mean,
        "posterior_var": var,
        "posterior_integral": float(integral),
    }


def _choose_recommendation(candidate_rows: list[dict[str, Any]], tail_sensitive: bool) -> dict[str, Any]:
    best_score = max(row["heldout_mean_logpdf"] for row in candidate_rows)
    near_ties = []
    for row in candidate_rows:
        tolerance = max(row["heldout_se_logpdf"], 1e-3)
        if best_score - row["heldout_mean_logpdf"] <= tolerance:
            near_ties.append(row)

    if tail_sensitive:
        ranked = sorted(
            near_ties,
            key=lambda row: (-row["tail_mean_logpdf"], row["complexity_rank"], -row["heldout_mean_logpdf"]),
        )
    else:
        ranked = sorted(
            near_ties,
            key=lambda row: (-row["heldout_mean_logpdf"], row["complexity_rank"], -row["tail_mean_logpdf"]),
        )
    return dict(ranked[0])


def _run_model_audit(spec: dict[str, Any]) -> dict[str, Any]:
    params = _default_params(spec["n"], spec["model_kw"])
    module = spec["module"]

    key = random.PRNGKey(spec["seed"])
    key, key_bench, key_obs = random.split(key, 3)
    benchmark = np.asarray(
        module.get_benchmark_mle_samples(key_bench, params, num_simulations=6000, verbose=False),
        dtype=float,
    )
    observed_data = np.asarray(module.sample_data(key_obs, params, loc=2.0), dtype=float)
    observed_mle = float(module.get_mle(observed_data, params))
    train, test = _split_samples(benchmark)

    candidate_rows = []
    for candidate in CANDIDATES:
        scores = _score_candidate(train, test, params, candidate["bw_method"])
        posterior = _posterior_summary(observed_mle, benchmark, params, candidate["bw_method"])
        candidate_rows.append(
            {
                "name": candidate["name"],
                "bw_method": candidate["bw_method"],
                "complexity_rank": candidate["complexity_rank"],
                **scores,
                **posterior,
            }
        )

    recommendation = _choose_recommendation(candidate_rows, tail_sensitive=spec["tail_sensitive"])
    recommended_name = recommendation["name"]
    runner_up = sorted(candidate_rows, key=lambda row: row["heldout_mean_logpdf"], reverse=True)[1]
    sensitivity = {
        "runner_up": runner_up["name"],
        "heldout_logpdf_gap": float(recommendation["heldout_mean_logpdf"] - runner_up["heldout_mean_logpdf"]),
        "posterior_mean_gap": float(abs(recommendation["posterior_mean"] - runner_up["posterior_mean"])),
        "posterior_var_rel_gap": float(
            abs(recommendation["posterior_var"] - runner_up["posterior_var"]) / max(abs(recommendation["posterior_var"]), 1e-12)
        ),
    }

    return {
        "model": spec["key"],
        "label": spec["label"],
        "n": spec["n"],
        "observed_mle": observed_mle,
        "statsmodels_installed": False,
        "candidates": candidate_rows,
        "recommended": recommendation,
        "sensitivity": sensitivity,
    }


def _chain_density(samples: np.ndarray, grid: np.ndarray, bw_method: str = "scott") -> np.ndarray:
    kde = stats.gaussian_kde(samples, bw_method=bw_method)
    return kde(grid)


def _generate_plot_payload(audit_payload: dict[str, Any]) -> dict[str, Any]:
    plot_payload: dict[str, Any] = {}
    audit_lookup = {entry["model"]: entry for entry in audit_payload["model_audits"]}
    for spec in MODEL_SPECS:
        if spec["plot_run"] is None:
            continue

        audit_entry = audit_lookup[spec["key"]]
        params = dict(spec["plot_run"]["base_params"])
        params["kde_bw_method"] = audit_entry["recommended"]["bw_method"]

        result = run_single_comparison(
            model=spec["model"],
            key=random.PRNGKey(0),
            n=spec["n"],
            mu_true=2.0,
            T_gibbs=spec["plot_run"]["T_gibbs"],
            T_baseline=spec["plot_run"]["T_baseline"],
            T_kde=spec["plot_run"]["T_kde"],
            T_fulldata=spec["plot_run"]["T_fulldata"],
            base_params=params,
            burnin=spec["plot_run"]["burnin"],
            verbose=False,
            seed_hint=0,
            **spec["model_kw"],
        )

        mu_gibbs = np.asarray(result["mu_chain_post_burnin"], dtype=float)
        mu_rattle = np.asarray(result["mu_chain_rattle_post_burnin"], dtype=float)
        mu_full = np.asarray(result["mu_chain_fulldata"][spec["plot_run"]["burnin"] :], dtype=float)
        lo = min(mu_gibbs.min(), mu_rattle.min(), mu_full.min(), result["kde_mean"]) - 0.5
        hi = max(mu_gibbs.max(), mu_rattle.max(), mu_full.max(), result["kde_mean"]) + 0.5
        grid = np.linspace(lo, hi, 600)

        plot_payload[spec["key"]] = {
            "model": spec["key"],
            "label": spec["label"],
            "n": spec["n"],
            "kde_bw_method": audit_entry["recommended"]["bw_method"],
            "kde_mean": float(result["kde_mean"]),
            "kde_variance": float(result["kde_variance"]),
            "grid": grid.tolist(),
            "gibbs_density": _chain_density(mu_gibbs, grid).tolist(),
            "rattle_density": _chain_density(mu_rattle, grid).tolist(),
            "full_data_density": _chain_density(mu_full, grid).tolist(),
            "kde_density": np.asarray(result["kde_posterior_pdf"](grid), dtype=float).tolist(),
        }
    return plot_payload


def run_kde_bandwidth_audit() -> dict[str, Any]:
    model_audits = [_run_model_audit(spec) for spec in MODEL_SPECS]
    payload = {
        "selection_rule": {
            "primary_score": "heldout_mean_logpdf on benchmark MLE samples",
            "tie_break": "within one standard-error of the best heldout score, prefer better tail score; if still tied, prefer lower-complexity backend",
            "notes": "statsmodels is not installed in this environment, so SJ_transform uses the fallback asinh-transform plus Silverman Gaussian KDE on the transformed scale.",
        },
        "model_audits": model_audits,
    }
    payload["plot_payload"] = _generate_plot_payload(payload)
    return payload


def main() -> None:
    payload = run_kde_bandwidth_audit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(OUT_JSON)


if __name__ == "__main__":
    main()
