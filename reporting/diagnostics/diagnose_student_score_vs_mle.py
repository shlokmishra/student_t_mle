"""Diagnose Student-t score-root constraints against the selected MLE convention.

This script is intentionally narrow: it checks whether Student latent states
sampled by Gibbs/RATTLE satisfy the score equation at mu_star and whether the
repo's deterministic get_mle selector returns mu_star for those same latent
states.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc_student import get_mle


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
    parser.add_argument("--loglik-gain-tol", type=float, default=1e-6)
    return parser.parse_args()


def latent_path(args: argparse.Namespace) -> Path:
    if args.latent_csv is not None:
        return args.latent_csv
    return args.chain_csv.parent / "latent_x_diagnostics.csv"


def x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if str(col).startswith("x_")]
    return sorted(cols, key=lambda col: int(str(col).split("_", 1)[1]))


def loglik_student(x: np.ndarray, mu: float, k: float) -> float:
    return float(np.sum(stats.t.logpdf(x, df=float(k), loc=float(mu), scale=1.0)))


def classify_group(group: pd.DataFrame) -> str:
    if group.empty:
        return "no latent diagnostics available"
    score_frac = float(group["score_near_zero"].mean())
    mle_frac = float(group["selected_mle_near_mu_star"].mean())
    if score_frac >= 0.95 and mle_frac < 0.95:
        return "score-root vs selected-MLE target mismatch"
    if mle_frac >= 0.95:
        return "sampler mixing / finite run issue"
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
        score = float(np.sum((x - mu_star) / (k + (x - mu_star) ** 2)))
        selected_mle = np.nan
        mle_error = ""
        try:
            selected_mle = float(get_mle(x, {"k": k, "n": n}))
        except Exception as exc:  # pragma: no cover - rare diagnostic path
            mle_error = f"{type(exc).__name__}: {exc}"
        loglik_at_mu_star = loglik_student(x, mu_star, k)
        loglik_at_selected_mle = loglik_student(x, selected_mle, k) if np.isfinite(selected_mle) else np.nan
        selected_delta = selected_mle - mu_star if np.isfinite(selected_mle) else np.nan
        loglik_gain = loglik_at_selected_mle - loglik_at_mu_star if np.isfinite(loglik_at_selected_mle) else np.nan
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
                "score_at_mu_star": score,
                "score_near_zero": bool(abs(score) <= args.score_tol),
                "mu_star_stationary": bool(abs(score) <= args.score_tol),
                "selected_mle": selected_mle,
                "selected_mle_minus_mu_star": selected_delta,
                "selected_mle_near_mu_star": bool(np.isfinite(selected_delta) and abs(selected_delta) <= args.mle_tol),
                "mu_star_selected_mle": bool(np.isfinite(selected_delta) and abs(selected_delta) <= args.mle_tol),
                "loglik_at_mu_star": loglik_at_mu_star,
                "loglik_at_selected_mle": loglik_at_selected_mle,
                "loglik_selected_minus_mu_star": loglik_gain,
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
                "fraction_selected_mle_near_mu_star": float(group["selected_mle_near_mu_star"].mean()),
                "target_mismatch_rate": float((group["score_near_zero"] & ~group["selected_mle_near_mu_star"]).mean()),
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


def markdown_report(args: argparse.Namespace, diagnostics: pd.DataFrame, summary: pd.DataFrame, warning: str = "") -> str:
    lines = ["# Student k=1,n=10 diagnostic", ""]
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
                "python scripts/run_cost_audit.py --models student_t --methods gibbs rattle --k-values 1 --n-values 10 --num-iterations 10000 --burn-in 2000 --seed 0 --out results/student_k1_n10_target_diag/ --save-latent-diagnostics --latent-diagnostic-thin 10",
                "```",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(["## Summary", "", "```text", summary.to_string(index=False), "```", ""])
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = args.out_dir / "student_score_vs_mle_diagnostics.csv"
    markdown_path = args.out_dir / "student_k1_n10_diagnostic.md"
    path = latent_path(args)

    warning = ""
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
    diagnostics.to_csv(diagnostics_csv, index=False)
    markdown_path.write_text(markdown_report(args, diagnostics, summary, warning), encoding="utf-8")
    print(summary.to_string(index=False) if not summary.empty else "No matching diagnostics rows.")
    print(f"wrote {diagnostics_csv}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
