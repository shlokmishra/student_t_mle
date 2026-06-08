"""Run one final-production diagnostic case on Grace.

The sampler calls are unchanged; this script only writes observational,
case-local diagnostics into a clean runset layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.random as random
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.cost_ledger import CostLedger
from scripts import run_cost_audit
from scripts.targeted_validation_config import find_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-config", type=Path, default=Path("configs/final_production_v1_cases.yaml"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/final_production_v1"))
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--diagnostic-thin", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= lag or lag <= 0:
        return np.nan
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def post_indices(length: int, burn_in: int, thin: int) -> np.ndarray:
    start = min(max(int(burn_in), 0), max(length - 1, 0))
    return np.arange(start, length, max(int(thin), 1), dtype=int)


def psi_values(model: str, x: np.ndarray, mu_star: float, k: float | None) -> np.ndarray:
    y = np.asarray(x, dtype=float) - float(mu_star)
    if model == "student_t":
        return y / (float(k) + y * y)
    if model == "logistic":
        return np.tanh(0.5 * y)
    return np.full_like(y, np.nan, dtype=float)


def constraint_residual(model: str, x: np.ndarray, mu_star: float, k: float | None) -> float:
    y = np.asarray(x, dtype=float) - float(mu_star)
    if model == "student_t":
        return float(np.sum(y / (float(k) + y * y)))
    if model == "logistic":
        return float(np.sum(np.tanh(0.5 * y)))
    if model == "laplace":
        return float(np.median(np.asarray(x, dtype=float)) - float(mu_star))
    return np.nan


def constraint_grad(model: str, x: np.ndarray, mu_star: float, k: float | None) -> np.ndarray:
    y = np.asarray(x, dtype=float) - float(mu_star)
    if model == "student_t":
        denom = float(k) + y * y
        return (float(k) - y * y) / (denom * denom)
    if model == "logistic":
        z = np.tanh(0.5 * y)
        return 0.5 * (1.0 - z * z)
    return np.full_like(y, np.nan, dtype=float)


def branch_labels(model: str, x: np.ndarray, mu_star: float, k: float | None) -> list[str]:
    if model != "student_t":
        return []
    threshold = np.sqrt(float(k))
    y = np.asarray(x, dtype=float) - float(mu_star)
    return ["lower" if abs(value) <= threshold else "upper" for value in y]


def tail_geometry_class(model: str, x: np.ndarray, mu_star: float, k: float | None) -> tuple[str, dict[str, float]]:
    y = np.asarray(x, dtype=float) - float(mu_star)
    abs_y = np.abs(y)
    if model == "student_t":
        threshold = np.sqrt(float(k))
        tail_fraction = float(np.mean(abs_y > threshold))
        extreme_fraction = float(np.mean(abs_y > 4.0 * threshold))
    elif model == "logistic":
        tail_fraction = float(np.mean(abs_y > 4.0))
        extreme_fraction = float(np.mean(abs_y > 8.0))
    else:
        tail_fraction = float(np.mean(abs_y > 3.0))
        extreme_fraction = float(np.mean(abs_y > 6.0))
    if extreme_fraction > 0.0:
        klass = "extreme_tail"
    elif tail_fraction >= 0.5:
        klass = "tail_dominated"
    elif tail_fraction > 0.0:
        klass = "mixed_tail"
    else:
        klass = "central"
    return klass, {
        "tail_fraction": tail_fraction,
        "extreme_tail_fraction": extreme_fraction,
        "x_min": float(np.min(y)),
        "x_max": float(np.max(y)),
        "x_mean": float(np.mean(y)),
        "x_sd": float(np.std(y)),
        "x_abs_max": float(np.max(abs_y)),
    }


def chain_rows(case: dict, chain: dict, burn_in: int, out_path: Path) -> list[dict]:
    mus = np.asarray(chain.get("mu_chain", []), dtype=float)
    return [
        {
            "case_id": case["case_id"],
            "model": case["model"],
            "method": case["method"],
            "k": np.nan if case.get("k") is None else float(case["k"]),
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "iteration": int(i),
            "mu": float(mu),
            "is_burn_in": bool(i < int(burn_in)),
            "source_file": str(out_path),
        }
        for i, mu in enumerate(mus)
    ]


def transition_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    mus = np.asarray(chain.get("mu_chain", []), dtype=float)
    if xs.ndim != 2 or mus.size == 0:
        return []
    rows = []
    model = str(case["model"])
    k = None if case.get("k") is None else float(case["k"])
    mu_star = 0.0
    prev_i = None
    for i in post_indices(xs.shape[0], burn_in, thin):
        x = xs[i]
        prev = xs[prev_i] if prev_i is not None else xs[max(i - 1, 0)]
        mu_prev = mus[prev_i] if prev_i is not None else mus[max(i - 1, 0)]
        psi = psi_values(model, x, mu_star, k)
        psi_prev = psi_values(model, prev, mu_star, k)
        pair_errors = [abs(float((psi[j] + psi[j + 1]) - (psi_prev[j] + psi_prev[j + 1]))) for j in range(0, len(psi) - 1, 2)]
        row = {
            "case_id": case["case_id"],
            "model": model,
            "method": case["method"],
            "k": np.nan if k is None else k,
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "iteration": int(i),
            "mu": float(mus[i]),
            "constraint_residual": constraint_residual(model, x, mu_star, k),
            "abs_constraint_residual": abs(constraint_residual(model, x, mu_star, k)),
            "abs_delta_mu": abs(float(mus[i] - mu_prev)),
            "ESJD_mu": float((mus[i] - mu_prev) ** 2),
            "movement_l2": float(np.linalg.norm(x - prev)),
            "pair_delta_old": float(np.nanmean([psi_prev[j] + psi_prev[j + 1] for j in range(0, len(psi_prev) - 1, 2)])) if len(psi_prev) > 1 else np.nan,
            "pair_delta_new": float(np.nanmean([psi[j] + psi[j + 1] for j in range(0, len(psi) - 1, 2)])) if len(psi) > 1 else np.nan,
            "abs_pair_delta_error": float(np.nanmax(pair_errors)) if pair_errors else np.nan,
            "student_inverse_branch_labels": "|".join(branch_labels(model, x, mu_star, k)),
            "source_file": str(out_path),
        }
        if model == "laplace":
            row.update(
                {
                    "median_minus_mu_star": float(np.median(x) - mu_star),
                    "count_below_mu_star": int(np.sum(x < mu_star)),
                    "count_equal_pinned_mu_star": int(np.sum(np.isclose(x, mu_star))),
                    "count_above_mu_star": int(np.sum(x > mu_star)),
                }
            )
        rows.append(row)
        prev_i = int(i)
    return rows


def latent_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    mus = np.asarray(chain.get("mu_chain", []), dtype=float)
    if xs.ndim != 2:
        return []
    rows = []
    model = str(case["model"])
    k = None if case.get("k") is None else float(case["k"])
    for i in post_indices(xs.shape[0], burn_in, thin):
        klass, metrics = tail_geometry_class(model, xs[i], 0.0, k)
        rows.append(
            {
                "case_id": case["case_id"],
                "model": model,
                "method": case["method"],
                "k": np.nan if k is None else k,
                "n": int(case["n"]),
                "seed": int(case["seed"]),
                "initialization": case["initialization"],
                "diagnostic_only": bool(case.get("diagnostic_only", False)),
                "iteration": int(i),
                "mu": float(mus[i]) if i < mus.size else np.nan,
                "latent_tail_geometry_class": klass,
                "constraint_residual": constraint_residual(model, xs[i], 0.0, k),
                "source_file": str(out_path),
                **metrics,
            }
        )
    return rows


def geometry_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    if xs.ndim != 2:
        return []
    rows = []
    previous = ""
    model = str(case["model"])
    k = None if case.get("k") is None else float(case["k"])
    for i in post_indices(xs.shape[0], burn_in, thin):
        klass, metrics = tail_geometry_class(model, xs[i], 0.0, k)
        grad = constraint_grad(model, xs[i], 0.0, k)
        gram = float(np.dot(grad, grad)) if np.all(np.isfinite(grad)) else np.nan
        rows.append(
            {
                "case_id": case["case_id"],
                "model": model,
                "method": case["method"],
                "k": np.nan if k is None else k,
                "n": int(case["n"]),
                "seed": int(case["seed"]),
                "initialization": case["initialization"],
                "diagnostic_only": bool(case.get("diagnostic_only", False)),
                "iteration": int(i),
                "latent_tail_geometry_class": klass,
                "previous_latent_tail_geometry_class": previous,
                "geometry_class_transition": f"{previous}->{klass}" if previous else "",
                "gram_value": gram,
                "source_file": str(out_path),
                **metrics,
            }
        )
        previous = klass
    return rows


def branch_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    if str(case["model"]) != "student_t" or xs.ndim != 2:
        return []
    counts: dict[str, int] = {"lower/lower": 0, "lower/upper": 0, "upper/lower": 0, "upper/upper": 0}
    switches = 0
    comparisons = 0
    previous_labels = None
    for i in post_indices(xs.shape[0], burn_in, thin):
        labels = branch_labels("student_t", xs[i], 0.0, float(case["k"]))
        for j in range(0, len(labels) - 1, 2):
            key = f"{labels[j]}/{labels[j + 1]}"
            counts[key] = counts.get(key, 0) + 1
        if previous_labels is not None:
            for old, new in zip(previous_labels, labels, strict=False):
                comparisons += 1
                switches += int(old != new)
        previous_labels = labels
    total_pairs = max(sum(counts.values()), 1)
    return [
        {
            "case_id": case["case_id"],
            "model": case["model"],
            "method": case["method"],
            "k": float(case["k"]),
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "branch_pair": key,
            "count": value,
            "frequency": float(value / total_pairs),
            "branch_switching_rate": float(switches / max(comparisons, 1)),
            "source_file": str(out_path),
        }
        for key, value in counts.items()
    ]


def rattle_energy_rows(case: dict, chain: dict, transition: list[dict], out_path: Path) -> list[dict]:
    diag = dict(chain.get("projection_diagnostics", {}))
    if str(case["method"]) != "rattle":
        return []
    movement = pd.DataFrame(transition)
    return [
        {
            "case_id": case["case_id"],
            "model": case["model"],
            "method": case["method"],
            "k": np.nan if case.get("k") is None else float(case["k"]),
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "position_constraint_residual": finite_float(diag.get("max_manifold_residual")),
            "position_constraint_residual_mean": finite_float(diag.get("mean_manifold_residual")),
            "tangent_residual_abs_grad_c_dot_p": finite_float(diag.get("max_tangent_residual", diag.get("max_projection_residual"))),
            "delta_H": finite_float(diag.get("delta_h_mean")),
            "delta_H_mean_abs": finite_float(diag.get("delta_h_mean_abs")),
            "delta_H_rms": finite_float(diag.get("delta_h_rms")),
            "delta_H_max_abs": finite_float(diag.get("delta_h_max_abs")),
            "gram_value": np.nan,
            "newton_iterations": finite_float(diag.get("projection_iterations_total")),
            "projection_failure_indicator": int(finite_float(diag.get("projection_failure_count", 0)) > 0),
            "reverse_check_failure_indicator": int(finite_float(diag.get("reverse_check_failure_count", 0)) > 0),
            "reverse_position_error": finite_float(diag.get("max_reverse_position_error")),
            "reverse_momentum_error": finite_float(diag.get("max_reverse_momentum_error")),
            "abs_delta_mu": float(movement["abs_delta_mu"].mean()) if "abs_delta_mu" in movement else np.nan,
            "ESJD_mu": float(movement["ESJD_mu"].mean()) if "ESJD_mu" in movement else np.nan,
            "source_file": str(out_path),
            **{key: value for key, value in diag.items() if np.isscalar(value)},
        }
    ]


def not_applicable_rattle_row(case: dict, out_path: Path) -> dict:
    return {
        "case_id": case["case_id"],
        "model": case["model"],
        "method": case["method"],
        "k": np.nan if case.get("k") is None else float(case["k"]),
        "n": int(case["n"]),
        "seed": int(case["seed"]),
        "initialization": case["initialization"],
        "diagnostic_only": bool(case.get("diagnostic_only", False)),
        "diagnostic_name": "rattle_energy",
        "diagnostic_applicable": False,
        "position_constraint_residual": np.nan,
        "position_constraint_residual_mean": np.nan,
        "tangent_residual_abs_grad_c_dot_p": np.nan,
        "delta_H": np.nan,
        "delta_H_mean_abs": np.nan,
        "delta_H_rms": np.nan,
        "delta_H_max_abs": np.nan,
        "gram_value": np.nan,
        "newton_iterations": np.nan,
        "projection_failure_indicator": np.nan,
        "reverse_check_failure_indicator": np.nan,
        "reverse_position_error": np.nan,
        "reverse_momentum_error": np.nan,
        "abs_delta_mu": np.nan,
        "ESJD_mu": np.nan,
        "source_file": str(out_path),
    }


def initialization_rows(case: dict, chain: dict, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    if xs.ndim != 2 or xs.shape[0] == 0:
        return []
    k = None if case.get("k") is None else float(case["k"])
    return [
        {
            "case_id": case["case_id"],
            "model": case["model"],
            "method": case["method"],
            "k": np.nan if k is None else k,
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "initial_constraint_residual": constraint_residual(str(case["model"]), xs[0], 0.0, k),
            "initial_x_min": float(np.min(xs[0])),
            "initial_x_max": float(np.max(xs[0])),
            "initial_x_sd": float(np.std(xs[0])),
            "source_file": str(out_path),
        }
    ]


def not_applicable_row(case: dict, diagnostic_name: str, out_path: Path) -> dict:
    return {
        "case_id": case["case_id"],
        "model": case["model"],
        "method": case["method"],
        "k": np.nan if case.get("k") is None else float(case["k"]),
        "n": int(case["n"]),
        "seed": int(case["seed"]),
        "initialization": case["initialization"],
        "diagnostic_only": bool(case.get("diagnostic_only", False)),
        "diagnostic_name": diagnostic_name,
        "diagnostic_applicable": False,
        "source_file": str(out_path),
    }


def run_args(case: dict, out_dir: Path, num_iterations: int, burn_in: int) -> SimpleNamespace:
    return SimpleNamespace(
        mu_star=0.0,
        num_iterations=int(num_iterations),
        burn_in=int(burn_in),
        seed=int(case["seed"]),
        out=out_dir,
        proposal_std_mu=0.3,
        proposal_std_z=0.02,
        prior_mean=0.0,
        prior_std=10.0,
        laplace_b=1.0,
        rattle_step_size=0.05,
        rattle_num_steps=2,
        rattle_proj_tol=1e-10,
        rattle_proj_max_iters=25,
        rattle_reverse_position_tol=5e-3,
        rattle_reverse_momentum_tol=5e-3,
        rattle_projection_mode="paper_fixed_direction",
        rattle_include_gram_correction=True,
        reverse_check=True,
        rattle_settings_json=None,
        run_status="final_production_v1",
        initialization=str(case["initialization"]),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    case = find_case(args.case_config, args.case_id)
    if args.seed is not None:
        case["seed"] = int(args.seed)
    num_iterations = int(args.num_iterations or case["num_iterations"])
    burn_in = int(args.burn_in or case["burn_in"])
    thin = int(args.diagnostic_thin or case["diagnostic_thin"])
    case_dir = args.out / f"case_{case['case_id']}"
    complete = case_dir / "run_metadata.json"
    if complete.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite completed case {case['case_id']}; pass --force to rerun.")
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "chain": case_dir / "chain_samples.csv",
        "posterior": case_dir / "posterior_summaries.csv",
        "ledger": case_dir / "cost_ledger.csv",
        "transition": case_dir / "transition_diagnostics.csv",
        "latent": case_dir / "latent_diagnostics.csv",
        "rattle": case_dir / "rattle_energy_diagnostics.csv",
        "branch": case_dir / "branch_diagnostics.csv",
        "geometry": case_dir / "geometry_diagnostics.csv",
        "init": case_dir / "initialization_diagnostics.csv",
    }
    key = random.PRNGKey(int(case["seed"]))
    k_value = np.nan if case.get("k") is None else float(case["k"])
    params = run_cost_audit.base_params(run_args(case, case_dir, num_iterations, burn_in), str(case["model"]), int(case["n"]), None if case.get("k") is None else float(case["k"]), {})
    runner = run_cost_audit.MODEL_MODULES[str(case["model"])][str(case["method"])]
    ledger = CostLedger(method=str(case["method"]), model=str(case["model"]), n=int(case["n"]), k=k_value, mu_star=0.0, seed=int(case["seed"]), iterations=num_iterations)
    meta = run_cost_audit._metadata(str(case["model"]), str(case["method"]), int(case["n"]))
    meta.update({"num_iterations": int(num_iterations), "burn_in": int(burn_in), "run_status": "final_production_v1"})
    ledger.counters.update(meta)
    ledger.start()
    if str(case["model"]) == "student_t":
        chain = runner(key, 0.0, params, verbose=False, cost_ledger=ledger)
    elif str(case["model"]) == "logistic" and str(case["method"]) == "rattle":
        chain = runner(key, 0.0, params, verbose=False, cost_ledger=ledger)
    else:
        ledger.set("iterations", int(num_iterations))
        chain = runner(key, 0.0, params, verbose=False)
    ledger.stop()

    summary_args = run_args(case, case_dir, num_iterations, burn_in)
    ledger_row, summary = run_cost_audit.summarize_chain(str(case["model"]), str(case["method"]), int(case["n"]), k_value, 0.0, int(case["seed"]), summary_args, chain, ledger)
    post = np.asarray(chain["mu_chain"], dtype=float)[min(burn_in, len(chain["mu_chain"]) - 1):]
    summary.update(
        {
            "case_id": case["case_id"],
            "initialization": case["initialization"],
            "diagnostic_only": bool(case.get("diagnostic_only", False)),
            "mu_autocorr_lag1": autocorr(post, 1),
            "mu_autocorr_lag5": autocorr(post, 5),
            "mu_autocorr_lag10": autocorr(post, 10),
            "source_file": str(paths["posterior"]),
        }
    )
    ledger_row.update({"case_id": case["case_id"], "initialization": case["initialization"], "diagnostic_only": bool(case.get("diagnostic_only", False)), "source_file": str(paths["ledger"])})

    chain_out = chain_rows(case, chain, burn_in, paths["chain"])
    transition = transition_rows(case, chain, burn_in, thin, paths["transition"])
    latent = latent_rows(case, chain, burn_in, thin, paths["latent"])
    rattle = rattle_energy_rows(case, chain, transition, paths["rattle"])
    branch = branch_rows(case, chain, burn_in, thin, paths["branch"])
    geometry = geometry_rows(case, chain, burn_in, thin, paths["geometry"])
    init = initialization_rows(case, chain, paths["init"])
    if not transition:
        transition = [not_applicable_row(case, "transition", paths["transition"])]
    if not latent:
        latent = [not_applicable_row(case, "latent", paths["latent"])]
    if not rattle:
        rattle = [not_applicable_rattle_row(case, paths["rattle"])]
    if not branch:
        branch = [not_applicable_row(case, "branch", paths["branch"])]
    if not geometry:
        geometry = [not_applicable_row(case, "geometry", paths["geometry"])]
    if not init:
        init = [not_applicable_row(case, "initialization", paths["init"])]

    write_csv(chain_out, paths["chain"])
    write_csv([summary], paths["posterior"])
    write_csv([ledger_row], paths["ledger"])
    write_csv(transition, paths["transition"])
    write_csv(latent, paths["latent"])
    write_csv(rattle, paths["rattle"])
    write_csv(branch, paths["branch"])
    write_csv(geometry, paths["geometry"])
    write_csv(init, paths["init"])
    metadata = {
        "case": case,
        "num_iterations": num_iterations,
        "burn_in": burn_in,
        "diagnostic_thin": thin,
        "full_latent_x_chain_saved": False,
        "files": {key: str(value) for key, value in paths.items()},
        "status": "completed",
    }
    complete.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"completed final production case {case['case_id']} -> {case_dir}")


if __name__ == "__main__":
    main()
