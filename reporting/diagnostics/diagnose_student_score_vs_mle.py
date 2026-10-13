"""Diagnose Student-t score-zero states against the global-MLE manifold.

The diagnostic can read saved latent states or generate Gibbs and RATTLE grids directly.
It classifies ``mu_star`` as an MLE by comparing its likelihood with the
certified production global selector, rather than using one wide Brent score
root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.random as random
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc.student import score_gibbs as loc_student
from models.loc.student import score_rattle as loc_student_rattle
from reporting.diagnostics.audit_student_global_mle import global_mle_dense, loglik, score


DEFAULT_CHAIN_CSV = Path("results/cost_audit/chain_samples.csv")
DEFAULT_SUMMARY_CSV = Path("results/cost_audit/posterior_summaries.csv")
DEFAULT_REFERENCE_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv")
DEFAULT_OUT_DIR = Path("results/analysis_report")


def _ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-csv", type=Path, default=DEFAULT_CHAIN_CSV)
    parser.add_argument("--posterior-summaries-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--latent-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k-values", type=_floats, default=[1.0])
    parser.add_argument("--n-values", type=_ints, default=[10])
    parser.add_argument("--model", default="student_t", choices=["student_t"])
    parser.add_argument("--score-tol", type=float, default=1e-6)
    parser.add_argument("--mle-tol", type=float, default=1e-3)
    parser.add_argument("--loglik-gain-tol", type=float, default=1e-8)
    parser.add_argument(
        "--run-gibbs-grid",
        action="store_true",
        help="Generate Gibbs chains for the requested k, n, and seed grid instead of reading latent CSV.",
    )
    parser.add_argument(
        "--run-rattle-grid",
        action="store_true",
        help="Generate RATTLE chains for the requested k, n, and seed grid instead of reading latent CSV.",
    )
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--chain-thin", type=int, default=1)
    parser.add_argument("--proposal-std-mu", type=float, default=0.2)
    parser.add_argument("--proposal-std-z", type=float, default=0.03)
    parser.add_argument("--prior-mean", type=float, default=0.0)
    parser.add_argument("--prior-std", type=float, default=10.0)
    parser.add_argument("--initialization", choices=["central", "tail_heavy", "random"], default="central")
    parser.add_argument("--rattle-step-size", type=float, default=0.05)
    parser.add_argument("--rattle-num-steps", type=int, default=2)
    parser.add_argument("--dense-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-validation-stride", type=int, default=200)
    parser.add_argument("--dense-points-per-interval", type=int, default=25)
    return parser.parse_args()


def latent_path(args: argparse.Namespace) -> Path:
    if args.latent_csv is not None:
        return args.latent_csv
    return args.chain_csv.parent / "latent_x_diagnostics.csv"


def x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("x_")]
    return sorted(cols, key=lambda col: int(str(col).split("_", 1)[1]))


def classify_group(group: pd.DataFrame) -> str:
    if group.empty:
        return "no latent diagnostics available"
    score_frac = float(group["score_near_zero"].mean())
    mle_frac = float(group["mu_star_is_global_mle"].mean())
    if score_frac >= 0.95 and bool((~group["mu_star_is_global_mle"]).any()):
        return "score-zero vs global-MLE target mismatch"
    if mle_frac >= 0.95:
        return "score-zero and global-MLE agree in sampled states"
    return "implementation or initialization issue"


def diagnostic_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    x_cols = x_columns(df)
    if not x_cols:
        return pd.DataFrame()
    rows = []
    focus = df[
        df["model"].astype(str).eq(args.model)
        & df["k"].astype(float).isin([float(k) for k in args.k_values])
        & df["n"].astype(int).isin([int(n) for n in args.n_values])
        & df["method"].astype(str).isin(["gibbs", "rattle"])
    ].copy()
    for record in focus.to_dict("records"):
        x = np.asarray([record[col] for col in x_cols if col in record and pd.notna(record[col])], dtype=float)
        k = float(record["k"])
        n = int(record["n"])
        mu_star = float(record["mu_star"])
        score_at_mu_star = float(np.sum((x - mu_star) / (k + (x - mu_star) ** 2)))
        selected_mle = selected_loglik = selected_score = np.nan
        num_multistarts = 0
        mle_error = ""
        try:
            selected_mle = float(loc_student.get_mle(x, {"k": k, "n": n}))
            selected_loglik = float(loglik(x, selected_mle, k))
            selected_score = abs(score(x, selected_mle, k))
        except Exception as exc:  # pragma: no cover - rare diagnostic path
            mle_error = f"{type(exc).__name__}: {exc}"
        loglik_at_mu_star = float(loglik(x, mu_star, k))
        loglik_at_selected_mle = float(selected_loglik) if np.isfinite(selected_loglik) else np.nan
        selected_delta = selected_mle - mu_star if np.isfinite(selected_mle) else np.nan
        loglik_gain = loglik_at_selected_mle - loglik_at_mu_star if np.isfinite(loglik_at_selected_mle) else np.nan
        dense_checked = False
        dense_agrees = True
        dense_minus_multistart_loglik = np.nan
        iteration = int(record.get("iteration", -1))
        validate_by_stride = args.dense_validation_stride > 0 and iteration % args.dense_validation_stride == 0
        if args.dense_validation and (validate_by_stride or loglik_gain > args.loglik_gain_tol):
            dense_checked = True
            dense_mle, dense_loglik = global_mle_dense(x, k, args.dense_points_per_interval)
            dense_minus_multistart_loglik = float(dense_loglik - loglik_at_selected_mle)
            dense_agrees = bool(dense_minus_multistart_loglik <= max(args.loglik_gain_tol, 1e-7))
            if dense_loglik > loglik_at_selected_mle:
                selected_mle = float(dense_mle)
                loglik_at_selected_mle = float(dense_loglik)
                selected_delta = selected_mle - mu_star
                loglik_gain = loglik_at_selected_mle - loglik_at_mu_star
        mu_star_is_global_mle = bool(np.isfinite(loglik_gain) and loglik_gain <= args.loglik_gain_tol)
        rows.append(
            {
                "model": record.get("model", args.model),
                "method": record.get("method", ""),
                "k": k,
                "n": n,
                "mu_star": mu_star,
                "seed": int(record.get("seed", 0)),
                "iteration": int(record.get("iteration", -1)),
                "mu": float(record.get("mu", np.nan)),
                "score_at_mu_star": score_at_mu_star,
                "score_near_zero": bool(abs(score_at_mu_star) <= args.score_tol),
                "mu_star_stationary": bool(abs(score_at_mu_star) <= args.score_tol),
                "selected_mle": selected_mle,
                "selected_mle_minus_mu_star": selected_delta,
                "selected_mle_near_mu_star": bool(np.isfinite(selected_delta) and abs(selected_delta) <= args.mle_tol),
                "mu_star_selected_mle": mu_star_is_global_mle,
                "mu_star_is_global_mle": mu_star_is_global_mle,
                "loglik_at_mu_star": loglik_at_mu_star,
                "loglik_at_selected_mle": loglik_at_selected_mle,
                "loglik_selected_minus_mu_star": loglik_gain,
                "selected_mle_score_abs": selected_score,
                "num_multistarts": num_multistarts,
                "dense_validation_checked": dense_checked,
                "dense_validation_agrees": dense_agrees,
                "dense_minus_multistart_loglik": dense_minus_multistart_loglik,
                "selected_mle_has_higher_loglik": bool(np.isfinite(loglik_gain) and loglik_gain > args.loglik_gain_tol),
                "mle_error": mle_error,
            }
        )
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    classifications = []
    for keys, group in diagnostics.groupby(["method", "k", "n"], dropna=False):
        method, k, n = keys
        classifications.append({"method": method, "k": k, "n": n, "classification": classify_group(group)})
    classifications = pd.DataFrame(classifications)
    return diagnostics.merge(classifications, on=["method", "k", "n"], how="left")


def summary_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in diagnostics.groupby(["method", "k", "n"], dropna=False):
        method, k, n = keys
        delta = group["selected_mle_minus_mu_star"].dropna()
        rows.append(
            {
                "method": method,
                "k": float(k),
                "n": int(n),
                "num_latent_draws": int(len(group)),
                "fraction_score_near_zero": float(group["score_near_zero"].mean()),
                "fraction_mu_star_is_global_mle": float(group["mu_star_is_global_mle"].mean()),
                "fraction_selected_mle_near_mu_star": float(group["selected_mle_near_mu_star"].mean()),
                "target_mismatch_rate": float((group["score_near_zero"] & ~group["mu_star_is_global_mle"]).mean()),
                "selected_mle_minus_mu_star_mean": float(delta.mean()) if not delta.empty else np.nan,
                "selected_mle_minus_mu_star_sd": float(delta.std(ddof=0)) if not delta.empty else np.nan,
                "selected_mle_minus_mu_star_q025": float(delta.quantile(0.025)) if not delta.empty else np.nan,
                "selected_mle_minus_mu_star_q50": float(delta.quantile(0.50)) if not delta.empty else np.nan,
                "selected_mle_minus_mu_star_q975": float(delta.quantile(0.975)) if not delta.empty else np.nan,
                "avg_loglik_selected_minus_mu_star": float(group["loglik_selected_minus_mu_star"].mean()),
                "fraction_selected_mle_higher_loglik": float(group["selected_mle_has_higher_loglik"].mean()),
                "classification": str(group["classification"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "n", "method"])


def seed_summary_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in diagnostics.groupby(["method", "k", "n", "seed"], dropna=False):
        method, k, n, seed = keys
        rows.append(
            {
                "method": method,
                "k": float(k),
                "n": int(n),
                "seed": int(seed),
                "num_latent_draws": int(len(group)),
                "fraction_score_near_zero": float(group["score_near_zero"].mean()),
                "fraction_mu_star_is_global_mle": float(group["mu_star_is_global_mle"].mean()),
                "target_mismatch_rate": float((group["score_near_zero"] & ~group["mu_star_is_global_mle"]).mean()),
                "max_loglik_gain_over_mu_star": float(group["loglik_selected_minus_mu_star"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "n", "seed", "method"])


def markdown_report(
    args: argparse.Namespace,
    diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    warning: str = "",
    seed_summary: pd.DataFrame | None = None,
) -> str:
    lines = ["# Student score-zero versus global-MLE diagnostic", ""]
    if warning:
        lines.extend(["## Status", "", warning, ""])
    lines.extend(
        [
            "## Inputs",
            "",
            f"- chain_csv: `{args.chain_csv}`",
            f"- latent_csv: `{latent_path(args)}`",
            f"- posterior_summaries_csv: `{args.posterior_summaries_csv}`",
            f"- reference_csv: `{args.reference_csv}`",
            f"- MLE selector: interval-certified root isolation, selected by actual Student log-likelihood",
            f"- global-MLE likelihood tolerance: `{args.loglik_gain_tol:g}`",
            f"- dense validation: every detected mismatch and every `{args.dense_validation_stride}`th state",
            "",
        ]
    )
    if summary.empty:
        lines.extend(
            [
                "## Result",
                "",
                "No latent x diagnostics were available. Re-run a targeted cost audit with:",
                "",
                "```bash",
                "The old score-benchmark runner is retired; see docs/research/student_selected_mle_gibbs_validation.md for current selected-MLE validation.",
                "```",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(["## Summary", "", "```text", summary.to_string(index=False), "```", ""])
    if seed_summary is not None and not seed_summary.empty:
        lines.extend(["## Per-seed summary", "", "```text", seed_summary.to_string(index=False), "```", ""])
    focus = summary[(summary["k"].eq(1.0)) & (summary["n"].eq(10))]
    if not focus.empty:
        classifications = sorted(set(focus["classification"].astype(str)))
        mismatch_rate = float(focus["target_mismatch_rate"].max())
        lines.extend(
            [
                "## Interpretation",
                "",
                f"- target_mismatch_rate: {mismatch_rate:.3f}",
                f"- recommendation: {', '.join(classifications)}",
                "",
            ]
        )
    if not diagnostics.empty:
        failures = diagnostics[diagnostics["mle_error"].astype(str).ne("")]
        if not failures.empty:
            lines.extend(["## MLE Selection Warnings", "", "```text", failures["mle_error"].value_counts().to_string(), "```", ""])
    return "\n".join(lines)


def run_sampler_grid(args: argparse.Namespace) -> pd.DataFrame:
    """Generate and immediately diagnose sampler chains to avoid a wide latent CSV."""
    if args.iterations < 1 or args.chain_thin < 1:
        raise ValueError("iterations and chain-thin must be positive")
    methods = []
    if args.run_gibbs_grid:
        methods.append("gibbs")
    if args.run_rattle_grid:
        methods.append("rattle")
    outputs = []
    for method in methods:
        for k in args.k_values:
            for n in args.n_values:
                for seed in args.seeds:
                    params = {
                        "k": float(k),
                        "n": int(n),
                        "num_iterations_T": int(args.iterations),
                        "proposal_std_mu": float(args.proposal_std_mu),
                        "proposal_std_z": float(args.proposal_std_z),
                        "prior_mean": float(args.prior_mean),
                        "prior_std": float(args.prior_std),
                        "gibbs_backend": "numba_full",
                        "initialization": str(args.initialization),
                        "initialization_seed": int(seed),
                        "store_x_chain": True,
                        "rattle_step_size": float(args.rattle_step_size),
                        "rattle_num_steps": int(args.rattle_num_steps),
                    }
                    if method == "gibbs":
                        chain = loc_student.run_gibbs(random.PRNGKey(int(seed)), 0.0, params, verbose=False)
                    else:
                        chain = loc_student_rattle.run_rattle(random.PRNGKey(int(seed)), 0.0, params, verbose=False)
                    states = np.asarray(chain["x_chain"], dtype=float)[:: int(args.chain_thin)]
                    iterations = np.arange(0, args.iterations + 1, args.chain_thin, dtype=int)
                    latent = pd.DataFrame(states, columns=[f"x_{index}" for index in range(int(n))])
                    latent["model"] = args.model
                    latent["method"] = method
                    latent["k"] = float(k)
                    latent["n"] = int(n)
                    latent["mu_star"] = 0.0
                    latent["seed"] = int(seed)
                    latent["iteration"] = iterations
                    latent["mu"] = np.asarray(chain["mu_chain"], dtype=float)[:: int(args.chain_thin)]
                    outputs.append(diagnostic_rows(latent, args))
                    print(
                        f"diagnosed {method}, k={float(k):g}, n={int(n)}, seed={int(seed)} ({len(states)} states)",
                        flush=True,
                    )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = args.out_dir / "student_score_vs_mle_diagnostics.csv"
    summary_csv = args.out_dir / "summary.csv"
    seed_summary_csv = args.out_dir / "summary_by_seed.csv"
    markdown_path = args.out_dir / "student_score_vs_global_mle_report.md"
    path = latent_path(args)

    warning = ""
    if args.run_gibbs_grid or args.run_rattle_grid:
        diagnostics = run_sampler_grid(args)
        summary = summary_table(diagnostics)
        seed_summary = seed_summary_table(diagnostics)
        diagnostics.to_csv(diagnostics_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        seed_summary.to_csv(seed_summary_csv, index=False)
        markdown_path.write_text(markdown_report(args, diagnostics, summary, seed_summary=seed_summary), encoding="utf-8")
        print(summary.to_string(index=False) if not summary.empty else "No matching diagnostics rows.")
        print(f"wrote {diagnostics_csv}")
        print(f"wrote {summary_csv}")
        print(f"wrote {seed_summary_csv}")
        print(f"wrote {markdown_path}")
        return
    if not path.exists():
        warning = "Latent target diagnostics require `x_chain` export. No latent_x_diagnostics.csv was found."
        pd.DataFrame().to_csv(diagnostics_csv, index=False)
        markdown_path.write_text(markdown_report(args, pd.DataFrame(), pd.DataFrame(), warning), encoding="utf-8")
        print(warning)
        print(f"wrote {diagnostics_csv}")
        print(f"wrote {markdown_path}")
        return

    latent = pd.read_csv(path)
    diagnostics = diagnostic_rows(latent, args)
    summary = summary_table(diagnostics)
    seed_summary = seed_summary_table(diagnostics)
    diagnostics.to_csv(diagnostics_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    seed_summary.to_csv(seed_summary_csv, index=False)
    markdown_path.write_text(markdown_report(args, diagnostics, summary, warning, seed_summary), encoding="utf-8")
    print(summary.to_string(index=False) if not summary.empty else "No matching diagnostics rows.")
    print(f"wrote {diagnostics_csv}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
