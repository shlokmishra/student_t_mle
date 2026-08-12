"""Run one targeted validation case with observational diagnostics."""

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
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from scripts import run_cost_audit
except ImportError:
    import run_cost_audit

try:
    from scripts.targeted_validation_config import find_case
except ModuleNotFoundError:
    from targeted_validation_config import find_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-config", type=Path, default=Path("configs/targeted_validation_cases.yaml"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/targeted_validation_runs"))
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--diagnostic-thin", type=int, default=None)
    parser.add_argument("--gibbs-backend", choices=["jax_loop", "jax_scan", "jax_scan_block_z", "numba"], default="jax_loop")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-transition-diagnostics", action="store_true")
    parser.add_argument("--save-latent-diagnostics", action="store_true")
    parser.add_argument("--save-rattle-energy-diagnostics", action="store_true")
    parser.add_argument("--save-branch-diagnostics", action="store_true")
    parser.add_argument("--save-initialization-diagnostics", action="store_true")
    parser.add_argument(
        "--save-full-latent-diagnostics",
        action="store_true",
        help="Write thinned latent_x_diagnostics.csv with x_0...x_{n-1} snapshots.",
    )
    parser.add_argument(
        "--full-latent-diagnostic-max-rows",
        type=int,
        default=0,
        help="Maximum full-latent snapshot rows per case; use 0 for no cap.",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def constraint_residual(model: str, x: np.ndarray, mu_star: float, k: float | None) -> float:
    y = np.asarray(x, dtype=float) - float(mu_star)
    if model == "student_t":
        return float(np.sum(y / (float(k) + y * y)))
    if model == "logistic":
        return float(np.sum(np.tanh(0.5 * y)))
    if model == "laplace":
        return float(np.median(np.asarray(x, dtype=float)) - float(mu_star))
    return np.nan


def psi_values(model: str, x: np.ndarray, mu_star: float, k: float | None) -> np.ndarray:
    y = np.asarray(x, dtype=float) - float(mu_star)
    if model == "student_t":
        return y / (float(k) + y * y)
    if model == "logistic":
        return np.tanh(0.5 * y)
    return np.full_like(y, np.nan, dtype=float)


def branch_labels(model: str, x: np.ndarray, mu_star: float, k: float | None) -> list[str]:
    if model != "student_t":
        return []
    y = np.asarray(x, dtype=float) - float(mu_star)
    threshold = np.sqrt(float(k))
    return ["lower" if abs(value) <= threshold else "upper" for value in y]


def post_indices(length: int, burn_in: int, thin: int) -> np.ndarray:
    start = min(max(int(burn_in), 0), max(length - 1, 0))
    return np.arange(start, length, max(int(thin), 1), dtype=int)


def transition_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    mus = np.asarray(chain.get("mu_chain", []), dtype=float)
    if xs.ndim != 2 or mus.size == 0:
        return []
    rows = []
    model = str(case["model"])
    k = None if case.get("k") is None else float(case["k"])
    mu_star = 0.0
    indices = post_indices(xs.shape[0], burn_in, thin)
    prev_i = None
    for i in indices:
        x = xs[i]
        prev = xs[prev_i] if prev_i is not None else xs[max(i - 1, 0)]
        mu_prev = mus[prev_i] if prev_i is not None else mus[max(i - 1, 0)]
        psi = psi_values(model, x, mu_star, k)
        psi_prev = psi_values(model, prev, mu_star, k)
        pair_errors = []
        for j in range(0, len(psi) - 1, 2):
            pair_errors.append(abs(float((psi[j] + psi[j + 1]) - (psi_prev[j] + psi_prev[j + 1]))))
        row = {
            "case_id": case["case_id"],
            "model": model,
            "method": case["method"],
            "k": k,
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "iteration": int(i),
            "mu": float(mus[i]),
            "constraint_residual": constraint_residual(model, x, mu_star, k),
            "abs_constraint_residual": abs(constraint_residual(model, x, mu_star, k)),
            "abs_delta_mu": abs(float(mus[i] - mu_prev)),
            "ESJD_mu": float((mus[i] - mu_prev) ** 2),
            "movement_l2": float(np.linalg.norm(x - prev)),
            "pair_delta_old": np.nanmean([psi_prev[j] + psi_prev[j + 1] for j in range(0, len(psi_prev) - 1, 2)]) if len(psi_prev) > 1 else np.nan,
            "pair_delta_new": np.nanmean([psi[j] + psi[j + 1] for j in range(0, len(psi) - 1, 2)]) if len(psi) > 1 else np.nan,
            "abs_pair_delta_error": float(np.nanmax(pair_errors)) if pair_errors else np.nan,
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
    if xs.ndim != 2:
        return []
    rows = []
    model = str(case["model"])
    k = None if case.get("k") is None else float(case["k"])
    for i in post_indices(xs.shape[0], burn_in, thin):
        x = xs[i]
        rows.append(
            {
                "case_id": case["case_id"],
                "model": model,
                "method": case["method"],
                "k": k,
                "n": int(case["n"]),
                "seed": int(case["seed"]),
                "initialization": case["initialization"],
                "iteration": int(i),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "x_mean": float(np.mean(x)),
                "x_sd": float(np.std(x)),
                "x_abs_max": float(np.max(np.abs(x))),
                "constraint_residual": constraint_residual(model, x, 0.0, k),
                "source_file": str(out_path),
            }
        )
    return rows


def branch_rows(case: dict, chain: dict, burn_in: int, thin: int, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    if str(case["model"]) != "student_t" or xs.ndim != 2:
        return []
    indices = post_indices(xs.shape[0], burn_in, thin)
    counts: dict[str, int] = {"lower/lower": 0, "lower/upper": 0, "upper/lower": 0, "upper/upper": 0}
    switches = 0
    comparisons = 0
    previous_labels = None
    for i in indices:
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
    row = {
        "case_id": case["case_id"],
        "model": case["model"],
        "method": case["method"],
        "k": np.nan if case.get("k") is None else float(case["k"]),
        "n": int(case["n"]),
        "seed": int(case["seed"]),
        "initialization": case["initialization"],
        "position_constraint_residual_max": finite_float(diag.get("max_manifold_residual")),
        "tangent_residual_max": finite_float(diag.get("max_tangent_residual", diag.get("max_projection_residual"))),
        "delta_H_mean": finite_float(diag.get("delta_h_mean")),
        "delta_H_mean_abs": finite_float(diag.get("delta_h_mean_abs")),
        "delta_H_rms": finite_float(diag.get("delta_h_rms")),
        "delta_H_max_abs": finite_float(diag.get("delta_h_max_abs")),
        "reverse_position_error": finite_float(diag.get("max_reverse_position_error")),
        "reverse_momentum_error": finite_float(diag.get("max_reverse_momentum_error")),
        "projection_failure_indicator": int(finite_float(diag.get("projection_failure_count", 0)) > 0),
        "reverse_check_failure_indicator": int(finite_float(diag.get("reverse_check_failure_count", 0)) > 0),
        "newton_iterations": finite_float(diag.get("projection_iterations_total")),
        "abs_delta_mu_mean": float(movement["abs_delta_mu"].mean()) if "abs_delta_mu" in movement else np.nan,
        "ESJD_mu_mean": float(movement["ESJD_mu"].mean()) if "ESJD_mu" in movement else np.nan,
        "movement_l2_mean": float(movement["movement_l2"].mean()) if "movement_l2" in movement else np.nan,
        "source_file": str(out_path),
    }
    for key, value in diag.items():
        if key not in row and np.isscalar(value):
            row[key] = value
    return [row]


def initialization_rows(case: dict, chain: dict, out_path: Path) -> list[dict]:
    xs = np.asarray(chain.get("x_chain", []), dtype=float)
    if xs.ndim != 2 or xs.shape[0] == 0:
        return []
    k = None if case.get("k") is None else float(case["k"])
    x0 = xs[0]
    return [
        {
            "case_id": case["case_id"],
            "model": case["model"],
            "method": case["method"],
            "k": k,
            "n": int(case["n"]),
            "seed": int(case["seed"]),
            "initialization": case["initialization"],
            "initial_constraint_residual": constraint_residual(str(case["model"]), x0, 0.0, k),
            "initial_x_min": float(np.min(x0)),
            "initial_x_max": float(np.max(x0)),
            "initial_x_sd": float(np.std(x0)),
            "source_file": str(out_path),
        }
    ]


def cost_args(case: dict, out_dir: Path, num_iterations: int, burn_in: int, gibbs_backend: str = "jax_loop") -> SimpleNamespace:
    n = int(case["n"])
    k = np.nan if case.get("k") is None else float(case["k"])
    return SimpleNamespace(
        models=[case["model"]],
        methods=[case["method"]],
        n_values=[n],
        laplace_n_values=[n],
        n_values_explicit=True,
        laplace_n_values_explicit=True,
        k_values=[] if case.get("k") is None else [float(case["k"])],
        k=None if case.get("k") is None else float(case["k"]),
        mu_star=0.0,
        num_iterations=int(num_iterations),
        burn_in=int(burn_in),
        seed=int(case["seed"]),
        out=out_dir,
        proposal_std_mu=0.3,
        proposal_std_z=0.02,
        gibbs_backend=str(gibbs_backend),
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
        run_status="targeted_validation",
        save_latent_diagnostics=False,
        latent_diagnostic_thin=10,
        latent_diagnostic_max_rows=0,
        initialization=str(case["initialization"]),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def not_applicable_diagnostic_row(case: dict, diagnostic_name: str, out_path: Path) -> dict:
    return {
        "case_id": case["case_id"],
        "model": case["model"],
        "method": case["method"],
        "k": np.nan if case.get("k") is None else float(case["k"]),
        "n": int(case["n"]),
        "seed": int(case["seed"]),
        "initialization": case["initialization"],
        "diagnostic_name": diagnostic_name,
        "diagnostic_applicable": False,
        "source_file": str(out_path),
    }


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

    run_args = cost_args(case, case_dir, num_iterations, burn_in, args.gibbs_backend)
    paths = {
        "chain": case_dir / "chain_samples.csv",
        "posterior": case_dir / "posterior_summaries.csv",
        "ledger": case_dir / "cost_ledger.csv",
        "transition": case_dir / "transition_diagnostics.csv",
        "latent": case_dir / "latent_diagnostics.csv",
        "rattle": case_dir / "rattle_energy_diagnostics.csv",
        "branch": case_dir / "branch_diagnostics.csv",
        "init": case_dir / "initialization_diagnostics.csv",
        "latent_x": case_dir / "latent_x_diagnostics.csv",
    }
    key = random.PRNGKey(int(case["seed"]))
    k_value = np.nan if case.get("k") is None else float(case["k"])
    spec = run_cost_audit.get_model_spec(str(case["model"]))
    if str(case["method"]) == "rattle" and not spec.supports_rattle:
        ledger_row, summary, diagnostic = run_cost_audit.not_applicable_row(
            str(case["model"]),
            str(case["method"]),
            int(case["n"]),
            k_value,
            run_args,
        )
        chain = {"mu_chain": np.asarray([]), "x_chain": np.asarray([])}
        rows = []
    else:
        params = run_cost_audit.base_params(run_args, str(case["model"]), int(case["n"]), None if case.get("k") is None else float(case["k"]), {})
        runner = run_cost_audit.MODEL_MODULES[str(case["model"])][str(case["method"])]
        ledger = run_cost_audit.CostLedger(
            method=str(case["method"]),
            model=str(case["model"]),
            n=int(case["n"]),
            k=k_value,
            mu_star=0.0,
            seed=int(case["seed"]),
            iterations=num_iterations,
        )
        meta = run_cost_audit._metadata(str(case["model"]), str(case["method"]), int(case["n"]))
        meta.update({"num_iterations": int(num_iterations), "burn_in": int(burn_in), "run_status": "targeted_validation"})
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
        ledger_row, summary = run_cost_audit.summarize_chain(
            str(case["model"]),
            str(case["method"]),
            int(case["n"]),
            k_value,
            0.0,
            int(case["seed"]),
            run_args,
            chain,
            ledger,
        )
        diagnostic = run_cost_audit.diagnostic_summary_row(ledger_row)
        rows = run_cost_audit.chain_rows(
            str(case["model"]),
            str(case["method"]),
            int(case["n"]),
            k_value,
            0.0,
            int(case["seed"]),
            burn_in,
            chain,
            meta,
        )

    for row in rows:
        row["initialization"] = case["initialization"]
        row["case_id"] = case["case_id"]
        row["source_file"] = str(paths["chain"])
    for row in [ledger_row, summary, diagnostic]:
        row["initialization"] = case["initialization"]
        row["case_id"] = case["case_id"]
    ledger_row["source_file"] = str(paths["ledger"])
    summary["source_file"] = str(paths["posterior"])

    transition = transition_rows(case, chain, burn_in, thin, paths["transition"])
    latent = latent_rows(case, chain, burn_in, thin, paths["latent"])
    full_latent = (
        run_cost_audit.latent_diagnostic_rows(
            str(case["model"]),
            str(case["method"]),
            int(case["n"]),
            k_value,
            0.0,
            int(case["seed"]),
            burn_in,
            chain,
            meta,
            thin,
            int(args.full_latent_diagnostic_max_rows),
        )
        if args.save_full_latent_diagnostics
        else []
    )
    branch = branch_rows(case, chain, burn_in, thin, paths["branch"])
    rattle = rattle_energy_rows(case, chain, transition, paths["rattle"])
    init = initialization_rows(case, chain, paths["init"])
    for row in full_latent:
        row["initialization"] = case["initialization"]
        row["case_id"] = case["case_id"]
        row["source_file"] = str(paths["latent_x"])
    if not transition:
        transition = [not_applicable_diagnostic_row(case, "transition", paths["transition"])]
    if not latent:
        latent = [not_applicable_diagnostic_row(case, "latent", paths["latent"])]
    if not branch:
        branch = [not_applicable_diagnostic_row(case, "branch", paths["branch"])]
    if not rattle:
        rattle = [not_applicable_diagnostic_row(case, "rattle_energy", paths["rattle"])]
    if not init:
        init = [not_applicable_diagnostic_row(case, "initialization", paths["init"])]

    write_csv(rows, paths["chain"])
    write_csv([summary], paths["posterior"])
    write_csv([ledger_row], paths["ledger"])
    write_csv(transition, paths["transition"])
    write_csv(latent, paths["latent"])
    if args.save_full_latent_diagnostics:
        write_csv(full_latent, paths["latent_x"])
    write_csv(rattle, paths["rattle"])
    write_csv(branch, paths["branch"])
    write_csv(init, paths["init"])
    metadata = {
        "case": case,
        "num_iterations": num_iterations,
        "burn_in": burn_in,
        "diagnostic_thin": thin,
        "files": {key: str(value) for key, value in paths.items()},
        "status": "completed",
    }
    complete.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"completed targeted validation case {case['case_id']} -> {case_dir}")


if __name__ == "__main__":
    main()
