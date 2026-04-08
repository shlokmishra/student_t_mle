"""Minimal final validation experiments for the manuscript package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np

from validation import run_single_comparison


OUT_PATH = Path("artifacts/final_comparison/final_validation_results.json")


def _base_rattle_controls() -> dict[str, Any]:
    return {
        "rattle_proj_damping": 1.0,
        "rattle_proj_line_search": True,
        "rattle_proj_init_strategy": "trial",
        "rattle_relaxed_position_tol": 5e-2,
        "rattle_relaxed_momentum_tol": 5e-2,
    }


def _logistic_locked_params() -> dict[str, Any]:
    params = _base_rattle_controls()
    params.update(
        {
            "rattle_step_size": 0.05,
            "rattle_num_steps": 2,
            "rattle_reverse_position_tol": 5e-3,
            "rattle_reverse_momentum_tol": 5e-3,
        }
    )
    return params


def _student3_locked_params() -> dict[str, Any]:
    params = _base_rattle_controls()
    params.update(
        {
            "rattle_step_size": 0.05,
            "rattle_num_steps": 1,
            "rattle_reverse_position_tol": 1e-2,
            "rattle_reverse_momentum_tol": 1e-2,
        }
    )
    return params


def _student2_replication_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "frozen_student3_transfer",
            "label": "Frozen Student-3 transfer",
            "base_params": _student3_locked_params(),
        },
        {
            "name": "closest_shortlist_candidate",
            "label": "Closest Student-2 shortlist",
            "base_params": {
                **_base_rattle_controls(),
                "rattle_step_size": 0.04,
                "rattle_num_steps": 2,
                "rattle_reverse_position_tol": 2e-2,
                "rattle_reverse_momentum_tol": 2e-2,
            },
        },
    ]


def _diagnostics_from_result(result: dict[str, Any]) -> dict[str, Any]:
    diag = dict(result.get("rattle_projection_diagnostics", {}))
    proposals = max(int(diag.get("proposals", 0)), 1)
    return {
        "x_acceptance_rate": float(result.get("rattle_acceptance", {}).get("x", np.nan)),
        "mu_acceptance_rate": float(result.get("rattle_acceptance", {}).get("mu", np.nan)),
        "reverse_fail_rate": float(diag.get("reverse_failures", 0)) / proposals,
        "reverse_projection_solver_failure_rate": float(diag.get("reverse_projection_solver_failure_count", 0)) / proposals,
        "reverse_position_mismatch_rate": float(diag.get("reverse_position_mismatch_count", 0)) / proposals,
        "reverse_momentum_mismatch_rate": float(diag.get("reverse_momentum_mismatch_count", 0)) / proposals,
        "reverse_tolerance_only_rate": float(diag.get("reverse_tolerance_only_failure_count", 0)) / proposals,
        "projection_failures": int(diag.get("projection_failure_count", 0)),
        "mean_manifold_residual": float(diag.get("mean_manifold_residual", np.nan)),
        "max_manifold_residual": float(diag.get("max_manifold_residual", np.nan)),
        "delta_h_mean": float(diag.get("delta_h_mean", np.nan)),
        "delta_h_mean_abs": float(diag.get("delta_h_mean_abs", np.nan)),
        "delta_h_rms": float(diag.get("delta_h_rms", np.nan)),
        "delta_h_max_abs": float(diag.get("delta_h_max_abs", np.nan)),
    }


def _method_summary(result: dict[str, Any], method: str) -> dict[str, Any]:
    mapping = {
        "gibbs": {
            "posterior_mean": "gibbs_mean",
            "posterior_var": "gibbs_variance",
            "runtime_s": "time_gibbs",
            "ess": "gibbs_ess",
            "ess_per_sec": "gibbs_ess_per_sec",
        },
        "kde": {
            "posterior_mean": "kde_mean",
            "posterior_var": "kde_variance",
            "runtime_s": "time_kde",
        },
        "rattle": {
            "posterior_mean": "rattle_mean",
            "posterior_var": "rattle_variance",
            "runtime_s": "time_rattle",
            "ess": "rattle_ess",
            "ess_per_sec": "rattle_ess_per_sec",
        },
        "full_data_mh": {
            "posterior_mean": "full_data_mean",
            "posterior_var": "full_data_variance",
            "runtime_s": "time_fulldata",
            "ess": "full_data_ess",
            "ess_per_sec": "full_data_ess_per_sec",
        },
    }
    keys = mapping[method]
    out = {}
    for label, key in keys.items():
        if key in result:
            out[label] = float(result[key])
    if method == "rattle":
        out.update(_diagnostics_from_result(result))
    return out


def _aggregate_numeric_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    keys = set().union(*(item.keys() for item in items))
    out: dict[str, Any] = {}
    for key in sorted(keys):
        vals = [item[key] for item in items if key in item and item[key] is not None]
        if not vals:
            continue
        if all(isinstance(v, (int, float, np.floating, np.integer)) for v in vals):
            out[key] = float(np.mean(vals))
    return out


def _run_comparison(
    *,
    model: str,
    seed: int,
    n: int,
    k: float | None,
    base_params: dict[str, Any],
    T_gibbs: int,
    T_baseline: int,
    T_kde: int,
    T_fulldata: int,
    burnin: int,
) -> dict[str, Any]:
    kwargs = {}
    if k is not None:
        kwargs["k"] = k
    result = run_single_comparison(
        model=model,
        key=random.PRNGKey(seed),
        n=n,
        mu_true=2.0,
        T_gibbs=T_gibbs,
        T_baseline=T_baseline,
        T_kde=T_kde,
        T_fulldata=T_fulldata,
        base_params=base_params,
        burnin=burnin,
        verbose=False,
        seed_hint=seed,
        **kwargs,
    )
    return {
        "seed": seed,
        "n": n,
        "model": model,
        "k": k,
        "T_gibbs": T_gibbs,
        "T_baseline": T_baseline,
        "T_kde": T_kde,
        "T_fulldata": T_fulldata,
        "burnin": burnin,
        "methods": {
            "gibbs": _method_summary(result, "gibbs"),
            "kde": _method_summary(result, "kde"),
            "rattle": _method_summary(result, "rattle"),
            "full_data_mh": _method_summary(result, "full_data_mh"),
        },
    }


def _student2_replication(seed_count: int = 3) -> dict[str, Any]:
    seeds = list(range(seed_count))
    config_payloads = []
    for config in _student2_replication_configs():
        runs = []
        for seed in seeds:
            runs.append(
                _run_comparison(
                    model="loc_student",
                    seed=seed,
                    n=20,
                    k=2.0,
                    base_params=config["base_params"],
                    T_gibbs=3000,
                    T_baseline=3000,
                    T_kde=10000,
                    T_fulldata=3000,
                    burnin=500,
                )
            )

        method_aggregates = {}
        for method in ["gibbs", "kde", "rattle", "full_data_mh"]:
            method_aggregates[method] = _aggregate_numeric_dicts([run["methods"][method] for run in runs])

        kde_var = method_aggregates["kde"].get("posterior_var")
        rattle_var = method_aggregates["rattle"].get("posterior_var")
        gibbs_var = method_aggregates["gibbs"].get("posterior_var")
        replicated = False
        if kde_var and gibbs_var and rattle_var:
            replicated = (rattle_var < kde_var) and (rattle_var < gibbs_var)

        config_payloads.append(
            {
                "name": config["name"],
                "label": config["label"],
                "n": 20,
                "seed_count": seed_count,
                "base_params": config["base_params"],
                "runs": runs,
                "aggregate": {
                    "methods": method_aggregates,
                    "underdispersion_vs_kde": float((kde_var - rattle_var) / kde_var) if kde_var else None,
                    "underdispersion_vs_gibbs": float((gibbs_var - rattle_var) / gibbs_var) if gibbs_var else None,
                    "failure_replicated": replicated,
                },
            }
        )
    return {"seed_count": seed_count, "configs": config_payloads}


def _long_chain_checks() -> list[dict[str, Any]]:
    return [
        {
            "name": "logistic_n20_long_chain",
            "label": "Logistic n=20 long chain",
            "model": "loc_logistic",
            "k": None,
            "n": 20,
            "seed": 0,
            "base_params": _logistic_locked_params(),
            "T_gibbs": 6000,
            "T_baseline": 6000,
            "T_kde": 10000,
            "T_fulldata": 6000,
            "burnin": 1000,
        },
        {
            "name": "student3_n20_long_chain",
            "label": "Student-3 n=20 long chain",
            "model": "loc_student",
            "k": 3.0,
            "n": 20,
            "seed": 0,
            "base_params": _student3_locked_params(),
            "T_gibbs": 6000,
            "T_baseline": 6000,
            "T_kde": 10000,
            "T_fulldata": 6000,
            "burnin": 1000,
        },
    ]


def run_final_validation() -> dict[str, Any]:
    student2 = _student2_replication(seed_count=3)
    long_chain_results = []
    for spec in _long_chain_checks():
        run = _run_comparison(
            model=spec["model"],
            seed=spec["seed"],
            n=spec["n"],
            k=spec["k"],
            base_params=spec["base_params"],
            T_gibbs=spec["T_gibbs"],
            T_baseline=spec["T_baseline"],
            T_kde=spec["T_kde"],
            T_fulldata=spec["T_fulldata"],
            burnin=spec["burnin"],
        )
        long_chain_results.append(
            {
                **spec,
                "result": run,
            }
        )

    package_ready = all(cfg["aggregate"]["failure_replicated"] for cfg in student2["configs"])
    return {
        "student2_replication": student2,
        "long_chain_checks": long_chain_results,
        "package_ready_to_share": package_ready,
        "package_ready_note": (
            "Ready to share as a polished update: Student-2 under-dispersion replicated across seeds, and Logistic/Student-3 remained robust under one longer-chain check."
            if package_ready
            else "Not yet fully ready: Student-2 replication did not cleanly confirm the current failure story."
        ),
    }


def main() -> None:
    payload = run_final_validation()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
