"""Run cost/sampler audits for Student-t, logistic, and Laplace location models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.random as random
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.cost_ledger import CostLedger
from models import loc_laplace, loc_logistic, loc_logistic_rattle, loc_student, loc_student_rattle
from models.model_registry import get_model_spec
from models.model_registry import LAPLACE_MEDIAN_INTERVAL_TARGET


MODEL_MODULES = {
    "student_t": {"gibbs": loc_student.run_gibbs, "rattle": loc_student_rattle.run_rattle},
    "logistic": {"gibbs": loc_logistic.run_gibbs, "rattle": loc_logistic_rattle.run_rattle},
    "laplace": {"gibbs": loc_laplace.run_gibbs},
}


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    n_values_explicit = "--n-values" in sys.argv
    laplace_n_values_explicit = "--laplace-n-values" in sys.argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["student_t"], choices=["student_t", "logistic", "laplace"])
    parser.add_argument("--methods", nargs="+", default=["gibbs", "rattle"], choices=["gibbs", "rattle"])
    parser.add_argument("--n-values", type=_ints, default=[10, 20, 50])
    parser.add_argument("--laplace-n-values", type=_ints, default=[11, 21, 51])
    parser.add_argument("--k-values", type=_floats, default=[2.0])
    parser.add_argument("--k", type=float, default=None, help="Backward-compatible single Student-t k.")
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/cost_audit/"))
    parser.add_argument("--proposal-std-mu", type=float, default=0.3)
    parser.add_argument("--proposal-std-z", type=float, default=0.02)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--laplace-b", type=float, default=1.0)
    parser.add_argument("--rattle-step-size", type=float, default=0.05)
    parser.add_argument("--rattle-num-steps", type=int, default=2)
    parser.add_argument("--rattle-proj-tol", type=float, default=1e-10)
    parser.add_argument("--rattle-proj-max-iters", type=int, default=25)
    parser.add_argument("--rattle-reverse-position-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-reverse-momentum-tol", type=float, default=5e-3)
    parser.add_argument("--rattle-projection-mode", choices=["paper_fixed_direction", "normal_newton_legacy"], default="paper_fixed_direction")
    parser.add_argument("--rattle-include-gram-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rattle-settings-json", type=Path, default=None)
    parser.add_argument("--run-status", default=None, help="Label rows as smoke, medium, full, tuning, or partial.")
    parser.add_argument("--initialization", choices=["central", "tail_heavy", "random"], default="central")
    parser.add_argument(
        "--save-latent-diagnostics",
        action="store_true",
        help="Write a thinned latent_x_diagnostics.csv for target diagnostics.",
    )
    parser.add_argument(
        "--latent-diagnostic-thin",
        type=int,
        default=10,
        help="Keep every Nth post-burn-in latent x draw when latent diagnostics are enabled.",
    )
    parser.add_argument(
        "--latent-diagnostic-max-rows",
        type=int,
        default=2000,
        help="Maximum latent diagnostic rows per model/method/n/k run; use 0 for no cap.",
    )
    args = parser.parse_args()
    args.n_values_explicit = n_values_explicit
    args.laplace_n_values_explicit = laplace_n_values_explicit
    return args


def effective_sample_size(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n <= 1:
        return float(n)
    centered = values - np.mean(values)
    var = float(np.dot(centered, centered) / n)
    if var <= 0 or not np.isfinite(var):
        return float(n)
    autocorr_sum = 0.0
    for lag in range(1, n):
        acov = float(np.dot(centered[:-lag], centered[lag:]) / n)
        rho = acov / var
        if rho <= 0:
            break
        autocorr_sum += rho
    return float(max(n / (1.0 + 2.0 * autocorr_sum), 1.0))


def infer_run_status(args: argparse.Namespace) -> str:
    if args.run_status:
        return str(args.run_status)
    out_text = str(args.out).lower()
    if "smoke" in out_text:
        return "smoke"
    if "medium" in out_text:
        return "medium"
    if "tuning" in out_text:
        return "tuning"
    n_values = set(map(int, args.n_values))
    laplace_n_values = set(map(int, getattr(args, "laplace_n_values", [])))
    if int(args.num_iterations) >= 10000 and n_values >= {10, 20, 50} and laplace_n_values >= {11, 21, 51}:
        return "full"
    return "partial"


def settings_key(model: str, k: float | None, n: int) -> str:
    if model == "student_t":
        return f"student_t:k={float(k):g}:n={int(n)}"
    return f"{model}:n={int(n)}"


def load_rattle_settings(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "by_key" in data and isinstance(data["by_key"], dict):
        return data["by_key"]
    if isinstance(data, dict):
        return data
    return {}


def base_params(args: argparse.Namespace, model: str, n: int, k: float | None, rattle_settings: dict[str, dict] | None = None) -> dict:
    params = {
        "n": int(n),
        "num_iterations_T": int(args.num_iterations),
        "proposal_std_mu": float(args.proposal_std_mu),
        "proposal_std_z": float(args.proposal_std_z),
        "prior_mean": float(args.prior_mean),
        "prior_std": float(args.prior_std),
        "rattle_step_size": float(args.rattle_step_size),
        "rattle_num_steps": int(args.rattle_num_steps),
        "rattle_proj_tol": float(args.rattle_proj_tol),
        "rattle_proj_max_iters": int(args.rattle_proj_max_iters),
        "rattle_reverse_position_tol": float(args.rattle_reverse_position_tol),
        "rattle_reverse_momentum_tol": float(args.rattle_reverse_momentum_tol),
        "rattle_projection_mode": str(args.rattle_projection_mode),
        "rattle_include_gram_correction": bool(args.rattle_include_gram_correction),
        "reverse_check": bool(args.reverse_check),
        "initialization": str(getattr(args, "initialization", "central")),
        "initialization_seed": int(getattr(args, "seed", 0)),
    }
    if model == "student_t":
        params["k"] = float(k)
    if model == "laplace":
        params["b"] = float(args.laplace_b)
    selected = (rattle_settings or {}).get(settings_key(model, k, n), {})
    if selected:
        if "rattle_step_size" in selected:
            params["rattle_step_size"] = float(selected["rattle_step_size"])
        if "rattle_num_steps" in selected:
            params["rattle_num_steps"] = int(selected["rattle_num_steps"])
    return params


def chain_rows(model: str, method: str, n: int, k: float, mu_star: float, seed: int, burn_in: int, chain: dict, meta: dict) -> list[dict]:
    mus = np.asarray(chain["mu_chain"], dtype=float)
    return [
        {
            **meta,
            "model": model,
            "method": method,
            "n": int(n),
            "k": float(k) if np.isfinite(k) else np.nan,
            "mu_star": float(mu_star),
            "seed": int(seed),
            "iteration": int(i),
            "mu": float(mu),
            "is_burn_in": bool(i < burn_in),
        }
        for i, mu in enumerate(mus)
    ]


def latent_diagnostic_rows(
    model: str,
    method: str,
    n: int,
    k: float,
    mu_star: float,
    seed: int,
    burn_in: int,
    chain: dict,
    meta: dict,
    thin: int,
    max_rows: int,
) -> list[dict]:
    if model != "student_t" or "x_chain" not in chain:
        return []
    xs = np.asarray(chain["x_chain"], dtype=float)
    mus = np.asarray(chain.get("mu_chain", np.full(xs.shape[0], np.nan)), dtype=float)
    if xs.ndim != 2 or xs.shape[0] == 0:
        return []
    start = min(max(int(burn_in), 0), xs.shape[0] - 1)
    step = max(int(thin), 1)
    indices = np.arange(start, xs.shape[0], step, dtype=int)
    cap = int(max_rows)
    if cap > 0 and indices.size > cap:
        indices = indices[:cap]
    rows = []
    for i in indices:
        row = {
            **meta,
            "model": model,
            "method": method,
            "n": int(n),
            "k": float(k) if np.isfinite(k) else np.nan,
            "mu_star": float(mu_star),
            "seed": int(seed),
            "iteration": int(i),
            "mu": float(mus[i]) if i < mus.size else np.nan,
            "is_burn_in": bool(i < burn_in),
        }
        for j, value in enumerate(xs[i]):
            row[f"x_{j}"] = float(value)
        rows.append(row)
    return rows


def _metadata(model: str, method: str, n: int | None = None) -> dict:
    spec = get_model_spec(model)
    target_description = spec.target_description
    mle_convention = spec.mle_convention
    if model == "laplace" and method == "gibbs":
        target = LAPLACE_MEDIAN_INTERVAL_TARGET if n is not None and int(n) % 2 == 0 else {
            "target_description": "deterministic_median_equals_mu_star",
            "mle_convention": "unique sample median pinned at mu_star for odd n",
        }
        target_description = target["target_description"]
        mle_convention = target["mle_convention"]
    if model == "laplace" and method == "rattle":
        target_description = LAPLACE_MEDIAN_INTERVAL_TARGET["target_description"]
        mle_convention = LAPLACE_MEDIAN_INTERVAL_TARGET["mle_convention"]
    return {
        "model": model,
        "supports_rattle": bool(spec.supports_rattle),
        "rattle_status": "applicable" if method != "rattle" or spec.supports_rattle else "not_applicable",
        "mle_convention": mle_convention,
        "target_description": target_description,
    }


def summarize_chain(model: str, method: str, n: int, k: float, mu_star: float, seed: int, args: argparse.Namespace, chain: dict, ledger: CostLedger) -> tuple[dict, dict]:
    mus = np.asarray(chain["mu_chain"], dtype=float)
    post = mus[min(int(args.burn_in), mus.size - 1) :]
    ess = effective_sample_size(post)
    ess_per_sec = ess / max(float(ledger.counters["wall_time_sec"]), 1e-12)
    acceptance_rate = float(chain.get("mu_acceptance_rate", chain.get("x_acceptance_rate", np.nan)))
    if method == "rattle":
        acceptance_rate = float(chain.get("x_acceptance_rate", acceptance_rate))
    meta = _metadata(model, method, n)
    meta.update({"num_iterations": int(args.num_iterations), "burn_in": int(args.burn_in), "run_status": infer_run_status(args)})
    ledger.counters.update(meta)
    ledger_row = ledger.output_row(ess_mu=ess, ess_per_sec=ess_per_sec, acceptance_rate=acceptance_rate)
    summary = {
        **meta,
        "method": method,
        "n": int(n),
        "k": float(k) if np.isfinite(k) else np.nan,
        "mu_star": float(mu_star),
        "seed": int(seed),
        "iterations": int(ledger.counters["iterations"]),
        "num_iterations": int(args.num_iterations),
        "burn_in": int(args.burn_in),
        "run_status": infer_run_status(args),
        "mean_mu": float(np.mean(post)) if post.size else np.nan,
        "var_mu": float(np.var(post)) if post.size else np.nan,
        "sd_mu": float(np.std(post)) if post.size else np.nan,
        "q025_mu": float(np.quantile(post, 0.025)) if post.size else np.nan,
        "q50_mu": float(np.quantile(post, 0.5)) if post.size else np.nan,
        "q975_mu": float(np.quantile(post, 0.975)) if post.size else np.nan,
        "ess_mu": ess,
        "ess_per_sec": ess_per_sec,
        "acceptance_rate": acceptance_rate,
        "projection_mode": ledger.counters.get("projection_mode", ""),
        "gram_correction_enabled": ledger.counters.get("gram_correction_enabled", False),
    }
    return ledger_row, summary


def diagnostic_summary_row(ledger_row: dict) -> dict:
    iterations = max(float(ledger_row.get("iterations", 0)), 1.0)
    ess = max(float(ledger_row.get("ess_mu", 0)), 1e-12)
    hmc_proposals = max(float(ledger_row.get("hmc_proposals", 0.0)), 1.0)
    projection_evals = float(ledger_row.get("projection_evals", 0.0))
    projection_failures = float(ledger_row.get("projection_failures", 0.0))
    reverse_failures = float(ledger_row.get("reverse_check_failures", 0.0))
    return {
        "model": ledger_row.get("model", ""),
        "method": ledger_row.get("method", ""),
        "n": ledger_row.get("n", np.nan),
        "k": ledger_row.get("k", np.nan),
        "mu_star": ledger_row.get("mu_star", np.nan),
        "seed": ledger_row.get("seed", np.nan),
        "ess_per_sec": ledger_row.get("ess_per_sec", np.nan),
        "cost_per_effective_sample_sec": float(ledger_row.get("wall_time_sec", 0.0)) / ess,
        "wall_time_per_iteration_sec": float(ledger_row.get("wall_time_sec", 0.0)) / iterations,
        "acceptance_rate": ledger_row.get("acceptance_rate", np.nan),
        "projection_failure_rate": projection_failures / max(projection_evals, 1.0),
        "reverse_check_failure_rate": reverse_failures / hmc_proposals,
        "student_logpdf_evals_per_iteration": float(ledger_row.get("student_logpdf_evals", 0.0)) / iterations,
        "student_grad_evals_per_iteration": float(ledger_row.get("student_grad_evals", 0.0)) / iterations,
        "projection_evals_per_iteration": projection_evals / iterations,
        "projection_mode": ledger_row.get("projection_mode", ""),
        "gram_correction_enabled": ledger_row.get("gram_correction_enabled", False),
        "supports_rattle": ledger_row.get("supports_rattle", False),
        "rattle_status": ledger_row.get("rattle_status", ""),
        "mle_convention": ledger_row.get("mle_convention", ""),
        "target_description": ledger_row.get("target_description", ""),
        "num_iterations": ledger_row.get("num_iterations", ledger_row.get("iterations", np.nan)),
        "burn_in": ledger_row.get("burn_in", np.nan),
        "run_status": ledger_row.get("run_status", ""),
        "source_file": ledger_row.get("source_file", ""),
    }


def not_applicable_row(model: str, method: str, n: int, k: float, args: argparse.Namespace) -> tuple[dict, dict, dict]:
    spec = get_model_spec(model)
    meta = _metadata(model, method, n)
    base = {
        **meta,
        "method": method,
        "n": int(n),
        "k": float(k) if np.isfinite(k) else np.nan,
        "mu_star": float(args.mu_star),
        "seed": int(args.seed),
        "iterations": 0,
        "num_iterations": int(args.num_iterations),
        "burn_in": int(args.burn_in),
        "run_status": infer_run_status(args),
        "wall_time_sec": 0.0,
        "projection_mode": "",
        "gram_correction_enabled": False,
        "ess_mu": np.nan,
        "ess_per_sec": np.nan,
        "acceptance_rate": np.nan,
    }
    return base, {**base, "burn_in": int(args.burn_in)}, diagnostic_summary_row(base)


def run_one(
    model: str,
    method: str,
    n: int,
    k: float,
    args: argparse.Namespace,
    seed: int,
    rattle_settings: dict[str, dict],
) -> tuple[list[dict], dict, dict, dict, list[dict]]:
    spec = get_model_spec(model)
    if method == "rattle" and not spec.supports_rattle:
        ledger_row, summary, diagnostic = not_applicable_row(model, method, n, k, args)
        return [], ledger_row, summary, diagnostic, []

    params = base_params(args, model, n, k, rattle_settings)
    key = random.PRNGKey(seed)
    ledger = CostLedger(method=method, model=model, n=n, k=k, mu_star=args.mu_star, seed=seed, iterations=args.num_iterations)
    meta = _metadata(model, method, n)
    meta.update({"num_iterations": int(args.num_iterations), "burn_in": int(args.burn_in), "run_status": infer_run_status(args)})
    ledger.counters.update(meta)
    ledger.start()
    runner = MODEL_MODULES[model][method]
    if model == "student_t":
        chain = runner(key, args.mu_star, params, verbose=False, cost_ledger=ledger)
    elif model == "logistic" and method == "rattle":
        chain = runner(key, args.mu_star, params, verbose=False, cost_ledger=ledger)
    else:
        ledger.set("iterations", int(args.num_iterations))
        chain = runner(key, args.mu_star, params, verbose=False)
    ledger.stop()
    ledger_row, summary = summarize_chain(model, method, n, k, args.mu_star, seed, args, chain, ledger)
    latent_rows = []
    if args.save_latent_diagnostics:
        latent_rows = latent_diagnostic_rows(
            model,
            method,
            n,
            k,
            args.mu_star,
            seed,
            args.burn_in,
            chain,
            meta,
            args.latent_diagnostic_thin,
            args.latent_diagnostic_max_rows,
        )
    return chain_rows(model, method, n, k, args.mu_star, seed, args.burn_in, chain, meta), ledger_row, summary, diagnostic_summary_row(ledger_row), latent_rows


def model_k_values(args: argparse.Namespace, model: str) -> list[float]:
    if model == "student_t":
        return [float(args.k)] if args.k is not None else [float(k) for k in args.k_values]
    return [np.nan]


def model_n_values(args: argparse.Namespace, model: str) -> list[int]:
    if model == "laplace" and (
        getattr(args, "laplace_n_values_explicit", False) or not getattr(args, "n_values_explicit", False)
    ):
        return [int(n) for n in args.laplace_n_values]
    return [int(n) for n in args.n_values]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rattle_settings = load_rattle_settings(args.rattle_settings_json)
    chain_rows_all = []
    latent_rows_all = []
    ledger_rows = []
    summary_rows = []
    diagnostic_rows = []
    for model in args.models:
        for method in args.methods:
            if method not in MODEL_MODULES.get(model, {}) and not (model == "laplace" and method == "rattle"):
                continue
            for k in model_k_values(args, model):
                for n in model_n_values(args, model):
                    rows, ledger_row, summary, diagnostic, latent_rows = run_one(model, method, int(n), float(k), args, int(args.seed), rattle_settings)
                    chain_rows_all.extend(rows)
                    latent_rows_all.extend(latent_rows)
                    ledger_rows.append(ledger_row)
                    summary_rows.append(summary)
                    diagnostic_rows.append(diagnostic)
                    print(f"completed model={model} method={method} n={n} k={k} status={ledger_row.get('rattle_status', '')}")

    paths = {
        "chain": args.out / "chain_samples.csv",
        "ledger": args.out / "cost_ledger.csv",
        "summary": args.out / "posterior_summaries.csv",
        "diagnostic": args.out / "diagnostic_summary.csv",
        "latent": args.out / "latent_x_diagnostics.csv",
    }
    for row in chain_rows_all:
        row["source_file"] = str(paths["chain"])
    for row in ledger_rows:
        row["source_file"] = str(paths["ledger"])
    for row in summary_rows:
        row["source_file"] = str(paths["summary"])
    for row in diagnostic_rows:
        row["source_file"] = str(paths["diagnostic"])
    for row in latent_rows_all:
        row["source_file"] = str(paths["latent"])

    pd.DataFrame(chain_rows_all).to_csv(paths["chain"], index=False)
    pd.DataFrame(ledger_rows).to_csv(paths["ledger"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False)
    pd.DataFrame(diagnostic_rows).to_csv(paths["diagnostic"], index=False)
    if args.save_latent_diagnostics:
        pd.DataFrame(latent_rows_all).to_csv(paths["latent"], index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
