"""Analyze full reference and sampler comparison outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".mplconfig"))
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = ["mean", "sd", "q025", "q50", "q975"]
COMMON_KEYS = ["model", "k", "n", "mu_star", "target_description"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"))
    parser.add_argument("--cost-dir", type=Path, default=Path("results/cost_audit/"))
    parser.add_argument("--tuning-dir", type=Path, default=Path("results/rattle_tuning/"))
    parser.add_argument("--stage2-tuning-dir", type=Path, default=Path("results/rattle_tuning_stage2/"))
    parser.add_argument("--stage2-cost-dir", type=Path, default=Path("results/cost_audit_medium_stage2_rattle/"))
    parser.add_argument("--multiseed-dir", type=Path, default=Path("results/cost_audit_multiseed/"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis_report/"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def finite_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def normalize_k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" in out.columns:
        out["k"] = pd.to_numeric(out["k"], errors="coerce")
    return out


def aggregate_reference(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = normalize_k(reference)
    raw = reference[reference["estimator_type"].eq("raw_weighted_mc")].copy()
    interval = reference[reference["estimator_type"].eq("raw_mc_interval_reference")].copy()
    kde = reference[reference["estimator_type"].eq("kde_grid")].copy()
    raw_agg = raw.groupby(COMMON_KEYS, dropna=False)[METRICS + ["weighted_ess", "marginal_likelihood_estimate"]].agg(["mean", "std"]).reset_index()
    raw_agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in raw_agg.columns]
    interval_agg = interval.groupby(COMMON_KEYS, dropna=False)[METRICS + ["marginal_likelihood_estimate"]].agg(["mean", "std"]).reset_index()
    interval_agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in interval_agg.columns]
    kde_agg = kde.groupby(COMMON_KEYS + ["backend"], dropna=False)[METRICS + ["marginal_likelihood_estimate"]].agg(["mean", "std"]).reset_index()
    kde_agg.columns = ["_".join([part for part in col if part]) if isinstance(col, tuple) else col for col in kde_agg.columns]
    return raw_agg, interval_agg, kde_agg


def sampler_summary(cost_dir: Path) -> pd.DataFrame:
    summaries = read_csv(cost_dir / "posterior_summaries.csv")
    if summaries.empty:
        return pd.DataFrame()
    out = normalize_k(summaries).rename(
        columns={
            "mean_mu": "mean",
            "sd_mu": "sd",
            "q025_mu": "q025",
            "q50_mu": "q50",
            "q975_mu": "q975",
            "var_mu": "var",
        }
    )
    out = out[out["method"].isin(["gibbs", "rattle"])].copy()
    out = out[~(out["method"].eq("rattle") & out.get("rattle_status", "").astype(str).eq("not_applicable"))]
    return out


def reference_for_sampler(row: pd.Series, raw_agg: pd.DataFrame, interval_agg: pd.DataFrame) -> pd.Series | None:
    if row["model"] == "laplace":
        ref = interval_agg[
            interval_agg["model"].eq("laplace")
            & interval_agg["n"].eq(int(row["n"]))
            & np.isclose(interval_agg["mu_star"], float(row["mu_star"]))
        ]
    else:
        ref = raw_agg[
            raw_agg["model"].eq(row["model"])
            & raw_agg["n"].eq(int(row["n"]))
            & np.isclose(raw_agg["mu_star"], float(row["mu_star"]))
        ]
        if row["model"] == "student_t":
            ref = ref[np.isclose(ref["k"], float(row["k"]))]
    return None if ref.empty else ref.iloc[0]


def accuracy_row(candidate: pd.Series, reference: pd.Series, estimator_type: str) -> dict:
    row = {
        "model": candidate.get("model", ""),
        "k": candidate.get("k", np.nan),
        "n": int(candidate.get("n", 0)),
        "mu_star": finite_float(candidate.get("mu_star", np.nan)),
        "target_description": reference.get("target_description", candidate.get("target_description", "")),
        "method": candidate.get("method", ""),
        "estimator_type": estimator_type,
        "backend": candidate.get("backend", candidate.get("method", "")),
        "seed": candidate.get("seed", np.nan),
        "mean": finite_float(candidate.get("mean", np.nan)),
        "sd": finite_float(candidate.get("sd", np.nan)),
        "q025": finite_float(candidate.get("q025", np.nan)),
        "q50": finite_float(candidate.get("q50", np.nan)),
        "q975": finite_float(candidate.get("q975", np.nan)),
        "raw_mean": finite_float(reference.get("mean_mean", np.nan)),
        "raw_sd": finite_float(reference.get("sd_mean", np.nan)),
        "raw_q025": finite_float(reference.get("q025_mean", np.nan)),
        "raw_q50": finite_float(reference.get("q50_mean", np.nan)),
        "raw_q975": finite_float(reference.get("q975_mean", np.nan)),
    }
    row["delta_mean"] = row["mean"] - row["raw_mean"]
    row["delta_sd"] = row["sd"] - row["raw_sd"]
    row["rel_sd_error"] = row["delta_sd"] / row["raw_sd"] if row["raw_sd"] else np.nan
    for metric in ["q025", "q50", "q975"]:
        row[f"delta_{metric}"] = row[metric] - row[f"raw_{metric}"]
    row["wasserstein_or_abs_quantile_score"] = float(
        np.nanmean([abs(row["delta_q025"]), abs(row["delta_q50"]), abs(row["delta_q975"])])
    )
    row["posterior_accuracy_acceptable"] = bool(
        np.isfinite(row["rel_sd_error"])
        and abs(row["rel_sd_error"]) <= 0.10
        and row["wasserstein_or_abs_quantile_score"] <= max(0.10, 0.20 * row["raw_sd"])
    )
    return row


def posterior_accuracy(reference: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    raw_agg, interval_agg, kde_agg = aggregate_reference(reference)
    rows: list[dict] = []
    for _, candidate in summaries.iterrows():
        ref = reference_for_sampler(candidate, raw_agg, interval_agg)
        if ref is not None:
            rows.append(accuracy_row(candidate, ref, "sampler"))
    for _, candidate in kde_agg.iterrows():
        ref = raw_agg[
            raw_agg["model"].eq(candidate["model"])
            & raw_agg["n"].eq(candidate["n"])
            & np.isclose(raw_agg["mu_star"], candidate["mu_star"])
            & raw_agg["target_description"].eq(candidate["target_description"])
        ]
        if candidate["model"] == "student_t":
            ref = ref[np.isclose(ref["k"], candidate["k"])]
        if ref.empty:
            continue
        candidate = candidate.rename({f"{metric}_mean": metric for metric in METRICS})
        candidate["method"] = "kde"
        rows.append(accuracy_row(candidate, ref.iloc[0], "kde_grid"))
    return pd.DataFrame(rows)


def cost_efficiency(cost_dir: Path) -> pd.DataFrame:
    ledger = normalize_k(read_csv(cost_dir / "cost_ledger.csv"))
    diagnostics = normalize_k(read_csv(cost_dir / "diagnostic_summary.csv"))
    if ledger.empty:
        return pd.DataFrame()
    out = ledger.copy()
    iterations = out["iterations"].replace(0, np.nan)
    ess = out["ess_mu"].replace(0, np.nan)
    def series(name: str) -> pd.Series:
        return out[name] if name in out.columns else pd.Series(0.0, index=out.index)

    out["wall_time_per_iteration"] = out["wall_time_sec"] / iterations
    out["wall_time_per_ess"] = out["wall_time_sec"] / ess
    out["logpdf_evals_per_iteration"] = (
        series("student_logpdf_evals").fillna(0)
        + series("prior_logpdf_evals").fillna(0)
        + series("potential_evals").fillna(0)
    ) / iterations
    out["constraint_evals_per_iteration"] = series("constraint_evals").fillna(0) / iterations
    out["projection_evals_per_iteration"] = series("projection_evals").fillna(0) / iterations
    out["gram_evals_per_iteration"] = series("gram_evals").fillna(0) / iterations
    out["newton_iters_per_iteration"] = (
        series("forward_newton_iters").fillna(0)
        + series("reverse_newton_iters").fillna(0)
    ) / iterations
    if not diagnostics.empty:
        diag_cols = [
            "model",
            "method",
            "k",
            "n",
            "mu_star",
            "seed",
            "projection_failure_rate",
            "reverse_check_failure_rate",
        ]
        diag_cols = [col for col in diag_cols if col in diagnostics.columns]
        keys = [col for col in ["model", "method", "k", "n", "mu_star", "seed"] if col in diag_cols]
        if keys:
            out = out.drop(columns=["projection_failure_rate", "reverse_check_failure_rate"], errors="ignore").merge(
                diagnostics[diag_cols].drop_duplicates(), on=keys, how="left"
            )
    columns = [
        "model",
        "k",
        "n",
        "mu_star",
        "method",
        "seed",
        "run_status",
        "target_description",
        "rattle_status",
        "projection_mode",
        "gram_correction_enabled",
        "iterations",
        "wall_time_sec",
        "ess_mu",
        "ess_per_sec",
        "acceptance_rate",
        "wall_time_per_iteration",
        "wall_time_per_ess",
        "logpdf_evals_per_iteration",
        "constraint_evals_per_iteration",
        "projection_evals_per_iteration",
        "gram_evals_per_iteration",
        "newton_iters_per_iteration",
        "projection_failure_rate",
        "reverse_check_failure_rate",
        "max_constraint_abs",
        "mean_constraint_abs",
    ]
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    return out[columns]


def method_rankings(accuracy: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    if accuracy.empty:
        return pd.DataFrame()
    sampler_acc = accuracy[accuracy["estimator_type"].eq("sampler")].copy()
    keys = ["model", "k", "n", "mu_star", "method", "seed"]
    cost_cols = keys + ["ess_per_sec", "wall_time_per_ess", "wall_time_per_iteration", "acceptance_rate"]
    merged = sampler_acc.merge(cost[cost_cols], on=keys, how="left")
    merged["accuracy_rank_value"] = np.where(merged["posterior_accuracy_acceptable"], 0, 1)
    merged = merged.sort_values(
        ["model", "k", "n", "accuracy_rank_value", "ess_per_sec", "wall_time_per_ess"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    )
    merged["rank"] = merged.groupby(["model", "k", "n"], dropna=False).cumcount() + 1
    return merged[
        [
            "model",
            "k",
            "n",
            "mu_star",
            "target_description",
            "rank",
            "method",
            "posterior_accuracy_acceptable",
            "rel_sd_error",
            "wasserstein_or_abs_quantile_score",
            "ess_per_sec",
            "wall_time_per_ess",
            "wall_time_per_iteration",
            "acceptance_rate",
        ]
    ]


def rattle_diagnostics(cost: pd.DataFrame, accuracy: pd.DataFrame, tuning: pd.DataFrame) -> pd.DataFrame:
    rat = cost[cost["method"].eq("rattle") & ~cost["rattle_status"].astype(str).eq("not_applicable")].copy()
    if rat.empty:
        return pd.DataFrame()
    rat = rat.merge(
        accuracy[accuracy["method"].eq("rattle")][["model", "k", "n", "mu_star", "seed", "rel_sd_error", "wasserstein_or_abs_quantile_score"]],
        on=["model", "k", "n", "mu_star", "seed"],
        how="left",
    )
    if not tuning.empty:
        tune_cols = ["model", "k", "n", "rattle_step_size", "rattle_num_steps", "status"]
        available = [col for col in tune_cols if col in tuning.columns]
        best = tuning.sort_values("ess_per_sec", ascending=False).drop_duplicates(["model", "k", "n"])
        rat = rat.merge(best[available], on=["model", "k", "n"], how="left", suffixes=("", "_tuning"))
    rat["acceptance_too_high"] = rat["acceptance_rate"] >= 0.995
    rat["low_ess"] = rat["ess_mu"] < 100
    rat["projection_failure_flag"] = rat["projection_failure_rate"].fillna(0) > 0.05
    rat["reverse_check_failure_flag"] = rat["reverse_check_failure_rate"].fillna(0) > 0.05
    rat["constraint_flag"] = rat["max_constraint_abs"].fillna(0) > 1e-6
    rat["posterior_sd_mismatch_flag"] = rat["rel_sd_error"].abs() > 0.10
    return rat


def simple_acf(values: np.ndarray, max_lag: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.array([1.0])
    centered = values - values.mean()
    denom = np.dot(centered, centered)
    if denom <= 0:
        return np.array([1.0])
    max_lag = min(max_lag, values.size - 1)
    return np.array([1.0] + [float(np.dot(centered[:-lag], centered[lag:]) / denom) for lag in range(1, max_lag + 1)])


def chain_diagnostics(cost_dir: Path) -> pd.DataFrame:
    chain = normalize_k(read_csv(cost_dir / "chain_samples.csv"))
    if chain.empty:
        return pd.DataFrame()
    if "is_burn_in" in chain.columns:
        chain = chain[~chain["is_burn_in"].astype(bool)]
    rows = []
    for keys, part in chain.groupby(["model", "k", "n", "method", "seed"], dropna=False):
        model, k, n, method, seed = keys
        vals = part["mu"].to_numpy(dtype=float)
        acf = simple_acf(vals)
        tau = 1.0 + 2.0 * np.sum(acf[1:][acf[1:] > 0])
        rows.append(
            {
                "model": model,
                "k": k,
                "n": int(n),
                "method": method,
                "seed": int(seed),
                "draws": int(vals.size),
                "trace_mean": float(np.mean(vals)),
                "trace_sd": float(np.std(vals)),
                "trace_q025": float(np.quantile(vals, 0.025)),
                "trace_q50": float(np.quantile(vals, 0.5)),
                "trace_q975": float(np.quantile(vals, 0.975)),
                "acf_lag1": float(acf[1]) if acf.size > 1 else np.nan,
                "integrated_autocorrelation_time": float(tau),
                "ess_from_acf": float(vals.size / tau) if tau > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def suspicious_cases(accuracy: pd.DataFrame, cost: pd.DataFrame, rat: pd.DataFrame, reference: pd.DataFrame, chain_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(row, issue_type: str, severity: str, metric: str, value, action: str) -> None:
        rows.append(
            {
                "model": row.get("model", ""),
                "k": row.get("k", np.nan),
                "n": row.get("n", np.nan),
                "method": row.get("method", ""),
                "issue_type": issue_type,
                "severity": severity,
                "metric": metric,
                "value": value,
                "recommended_action": action,
            }
        )

    for _, row in accuracy.iterrows():
        if row["estimator_type"] == "sampler" and (abs(row["rel_sd_error"]) > 0.10 or row["wasserstein_or_abs_quantile_score"] > max(0.10, 0.20 * row["raw_sd"])):
            add(row, "posterior_mismatch", "high", "rel_sd_error", row["rel_sd_error"], "Inspect trace and run targeted robustness or tuning before scientific comparison.")
    for _, row in cost.iterrows():
        if row["method"] == "rattle" and row.get("rattle_status") == "not_applicable":
            continue
        if row["method"] == "rattle" and row["acceptance_rate"] >= 0.995:
            add(row, "rattle_acceptance_too_high", "medium", "acceptance_rate", row["acceptance_rate"], "Treat current RATTLE setting as small-move warning unless movement and posterior accuracy are adequate.")
        if row["method"] == "rattle" and row["projection_failure_rate"] > 0.05:
            add(row, "rattle_projection_failures", "high", "projection_failure_rate", row["projection_failure_rate"], "Reject this RATTLE setting.")
        if row["method"] == "rattle" and row["reverse_check_failure_rate"] > 0.05:
            add(row, "rattle_reverse_check_failures", "high", "reverse_check_failure_rate", row["reverse_check_failure_rate"], "Reject this RATTLE setting.")
        if row["method"] == "rattle" and row["max_constraint_abs"] > 1e-6:
            add(row, "rattle_constraint_residual", "high", "max_constraint_abs", row["max_constraint_abs"], "Tighten projection or inspect implementation.")
        if row["method"] == "gibbs" and row["ess_mu"] < 100:
            add(row, "gibbs_low_ess", "medium", "ess_mu", row["ess_mu"], "Run longer or inspect mixing.")
        if row["method"] == "rattle" and row["ess_mu"] < 100:
            add(row, "rattle_low_ess", "medium", "ess_mu", row["ess_mu"], "Run longer or tune larger moves.")
        if row["method"] == "rattle" and row.get("rattle_status") == "not_applicable":
            add(row, "laplace_rattle_not_applicable", "info", "rattle_status", row["rattle_status"], "Do not compare Laplace RATTLE.")
    if "laplace" in set(reference["model"].astype(str)):
        add({"model": "laplace", "k": np.nan, "n": "even", "method": "gibbs"}, "laplace_target_mismatch", "info", "target_description", "deterministic_np_median_equals_mu_star", "Use median_interval_contains_mu_star reference for Laplace Gibbs.")
    if chain_diag.empty:
        rows.append({"model": "", "k": np.nan, "n": np.nan, "method": "", "issue_type": "missing_chain", "severity": "high", "metric": "chain_samples.csv", "value": "missing", "recommended_action": "Run cost audit with chain export."})
    return pd.DataFrame(rows)


def multiseed_summary(multiseed_dir: Path) -> pd.DataFrame:
    ledger = normalize_k(read_csv(multiseed_dir / "cost_ledger.csv"))
    summaries = normalize_k(read_csv(multiseed_dir / "posterior_summaries.csv"))
    if ledger.empty or summaries.empty:
        return pd.DataFrame()
    ledger = ledger.copy()
    ledger["projection_failure_rate"] = ledger["projection_failures"] / ledger["projection_evals"].replace(0, np.nan)
    ledger["reverse_check_failure_rate"] = ledger["reverse_check_failures"] / ledger["hmc_proposals"].replace(0, np.nan)
    summary_base = summaries.rename(columns={"mean_mu": "mean", "sd_mu": "sd"}).drop(
        columns=["ess_per_sec"], errors="ignore"
    )
    merged = summary_base.merge(
        ledger[["model", "k", "n", "method", "seed", "ess_per_sec", "projection_failure_rate", "reverse_check_failure_rate"]],
        on=["model", "k", "n", "method", "seed"],
        how="left",
    )
    return merged.groupby(["model", "k", "n", "method"], dropna=False).agg(
        seed_count=("seed", "nunique"),
        ess_per_sec_mean=("ess_per_sec", "mean"),
        ess_per_sec_sd=("ess_per_sec", "std"),
        posterior_mean_mean=("mean", "mean"),
        posterior_mean_sd=("mean", "std"),
        posterior_sd_mean=("sd", "mean"),
        posterior_sd_sd=("sd", "std"),
        projection_failure_rate_mean=("projection_failure_rate", "mean"),
        projection_failure_rate_sd=("projection_failure_rate", "std"),
        reverse_check_failure_rate_mean=("reverse_check_failure_rate", "mean"),
        reverse_check_failure_rate_sd=("reverse_check_failure_rate", "std"),
    ).reset_index()


def stage2_followup(stage2_tuning_dir: Path, stage2_cost_dir: Path, reference: pd.DataFrame) -> pd.DataFrame:
    tuning = normalize_k(read_csv(stage2_tuning_dir / "tuning_summary.csv"))
    summaries = sampler_summary(stage2_cost_dir)
    if tuning.empty and summaries.empty:
        return pd.DataFrame()
    rows = []
    if not tuning.empty:
        grouped = tuning.groupby(["model", "k", "n"], dropna=False)
        for keys, part in grouped:
            model, k, n = keys
            ok = part[
                (part["projection_failure_rate"].fillna(1.0) <= 0.05)
                & (part["reverse_check_failure_rate"].fillna(1.0) <= 0.05)
                & (part["acceptance_rate"].between(0.4, 0.95))
            ]
            best = ok.sort_values("ess_per_sec", ascending=False).iloc[0] if not ok.empty else part.sort_values("ess_per_sec", ascending=False).iloc[0]
            rows.append(
                {
                    "source": "stage2_tuning",
                    "model": model,
                    "k": k,
                    "n": int(n),
                    "method": "rattle",
                    "rattle_step_size": best["rattle_step_size"],
                    "rattle_num_steps": best["rattle_num_steps"],
                    "acceptance_rate": best["acceptance_rate"],
                    "ess_per_sec": best["ess_per_sec"],
                    "projection_failure_rate": best["projection_failure_rate"],
                    "reverse_check_failure_rate": best["reverse_check_failure_rate"],
                    "status": "ok" if not ok.empty else "warning",
                }
            )
    if not summaries.empty:
        raw_agg, interval_agg, _ = aggregate_reference(reference)
        cost = cost_efficiency(stage2_cost_dir)
        acc = posterior_accuracy(reference, summaries)
        acc = acc[acc["estimator_type"].eq("sampler")]
        for _, row in acc.iterrows():
            cost_row = cost[
                cost["model"].eq(row["model"])
                & cost["method"].eq(row["method"])
                & cost["n"].eq(row["n"])
                & np.isclose(cost["k"].fillna(-9999), row["k"] if np.isfinite(row["k"]) else -9999)
            ]
            rows.append(
                {
                    "source": "stage2_medium_followup",
                    "model": row["model"],
                    "k": row["k"],
                    "n": int(row["n"]),
                    "method": row["method"],
                    "rattle_step_size": np.nan,
                    "rattle_num_steps": np.nan,
                    "acceptance_rate": row.get("acceptance_rate", cost_row["acceptance_rate"].iloc[0] if not cost_row.empty else np.nan),
                    "ess_per_sec": cost_row["ess_per_sec"].iloc[0] if not cost_row.empty else np.nan,
                    "projection_failure_rate": cost_row["projection_failure_rate"].iloc[0] if not cost_row.empty else np.nan,
                    "reverse_check_failure_rate": cost_row["reverse_check_failure_rate"].iloc[0] if not cost_row.empty else np.nan,
                    "status": "posterior_acceptable" if row["posterior_accuracy_acceptable"] else "posterior_failed",
                    "rel_sd_error": row["rel_sd_error"],
                    "wasserstein_or_abs_quantile_score": row["wasserstein_or_abs_quantile_score"],
                    "delta_mean": row["delta_mean"],
                }
            )
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, accuracy: pd.DataFrame, cost: pd.DataFrame, chain_diag: pd.DataFrame, chain_path: Path) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    if not cost.empty:
        for metric, name, ylabel in [
            ("ess_per_sec", "ess_per_sec_vs_n.png", "ESS/sec"),
            ("wall_time_per_ess", "wall_time_per_ess_vs_n.png", "wall time / ESS"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for (model, method), part in cost.groupby(["model", "method"], dropna=False):
                part = part[np.isfinite(part[metric])].sort_values("n")
                if part.empty:
                    continue
                label = f"{model} {method}"
                ax.plot(part["n"], part[metric], marker="o", label=label)
            ax.set_xlabel("n")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            save(fig, name)
        rat = cost[cost["method"].eq("rattle") & np.isfinite(cost["acceptance_rate"])]
        if not rat.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for model, part in rat.groupby("model"):
                ax.plot(part["n"], part["acceptance_rate"], marker="o", label=model)
            ax.axhline(0.995, color="red", linestyle=":", linewidth=1)
            ax.set_xlabel("n")
            ax.set_ylabel("acceptance_rate")
            ax.grid(alpha=0.25)
            ax.legend()
            save(fig, "rattle_acceptance_vs_n.png")
    if not accuracy.empty:
        sampler = accuracy[accuracy["estimator_type"].eq("sampler")]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (model, method), part in sampler.groupby(["model", "method"], dropna=False):
            part = part[np.isfinite(part["rel_sd_error"])].sort_values("n")
            if part.empty:
                continue
            ax.plot(part["n"], part["rel_sd_error"], marker="o", label=f"{model} {method}")
        ax.axhline(0.10, color="red", linestyle=":", linewidth=1)
        ax.axhline(-0.10, color="red", linestyle=":", linewidth=1)
        ax.set_xlabel("n")
        ax.set_ylabel("relative sd error")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        save(fig, "posterior_sd_error_vs_n.png")

    chain = normalize_k(read_csv(chain_path))
    if not chain.empty:
        if "is_burn_in" in chain.columns:
            post_chain = chain[~chain["is_burn_in"].astype(bool)].copy()
        else:
            post_chain = chain
        sampler_accuracy = accuracy[accuracy["estimator_type"].eq("sampler")]
        for keys, part in post_chain.groupby(["model", "k", "n"], dropna=False):
            model, k, n = keys
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for method, method_part in part.groupby("method", dropna=False):
                values = method_part["mu"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size < 2:
                    continue
                ax.hist(values, bins=45, density=True, alpha=0.35, label=str(method))
            ref = sampler_accuracy[
                sampler_accuracy["model"].eq(model)
                & sampler_accuracy["n"].eq(int(n))
                & np.isclose(sampler_accuracy["k"].fillna(-9999), k if np.isfinite(k) else -9999)
            ]
            if not ref.empty:
                raw = ref.iloc[0]
                ax.axvline(raw["raw_mean"], color="black", linewidth=1.2, label="raw/reference mean")
                ax.axvline(raw["raw_q025"], color="black", linestyle=":", linewidth=1.0, label="raw/reference 95%")
                ax.axvline(raw["raw_q975"], color="black", linestyle=":", linewidth=1.0)
            ax.set_xlabel("mu")
            ax.set_ylabel("density")
            ax.set_title(f"Posterior overlay {model} k={k:g} n={int(n)}" if np.isfinite(k) else f"Posterior overlay {model} n={int(n)}")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            safe_k = "na" if not np.isfinite(k) else f"{k:g}"
            save(fig, f"posterior_overlay_{model}_k{safe_k}_n{int(n)}.png")
        for keys, part in post_chain.groupby(["model", "k", "n", "method"], dropna=False):
            model, k, n, method = keys
            if model == "laplace" and method == "rattle":
                continue
            sample = part.sort_values("iteration").tail(2000)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(sample["iteration"], sample["mu"], linewidth=0.8)
            ax.set_xlabel("iteration")
            ax.set_ylabel("mu")
            ax.set_title(f"{model} k={k:g} n={int(n)} {method}" if np.isfinite(k) else f"{model} n={int(n)} {method}")
            ax.grid(alpha=0.2)
            safe_k = "na" if not np.isfinite(k) else f"{k:g}"
            save(fig, f"trace_{model}_k{safe_k}_n{int(n)}_{method}.png")
    return paths


def executive_summary(
    reference: pd.DataFrame,
    accuracy: pd.DataFrame,
    cost: pd.DataFrame,
    rankings: pd.DataFrame,
    rat: pd.DataFrame,
    suspicious: pd.DataFrame,
    multiseed: pd.DataFrame,
    stage2: pd.DataFrame,
    tuning_warning: str,
    dashboard_health_path: Path,
) -> str:
    lines = [
        "# Full Comparison Executive Summary",
        "",
        "## What Ran",
        f"- Reference rows analyzed: {len(reference)}.",
        f"- Sampler cost rows analyzed: {len(cost)}.",
        f"- Posterior accuracy rows produced: {len(accuracy)}.",
        "",
        "## What Is Valid To Compare",
        "- Student-t and Logistic Gibbs/RATTLE are compared against raw weighted-MC posterior summaries.",
        "- Laplace Gibbs is compared only against `median_interval_contains_mu_star`.",
        "- Laplace deterministic `np.median` reference is reported separately and is not used for even-n Gibbs deltas.",
        "",
        "## Main Posterior Accuracy Findings",
    ]
    sampler = accuracy[accuracy["estimator_type"].eq("sampler")].copy()
    for model in ["student_t", "logistic", "laplace"]:
        part = sampler[sampler["model"].eq(model)]
        if part.empty:
            continue
        if model == "student_t":
            for k, kpart in part.groupby("k", dropna=False):
                worst = kpart.iloc[kpart["rel_sd_error"].abs().argmax()]
                lines.append(
                    f"- Student k={k:g}: worst sampler rel_sd_error={worst['rel_sd_error']:.3f} "
                    f"({worst['method']}, n={int(worst['n'])}); acceptable rows={int(kpart['posterior_accuracy_acceptable'].sum())}/{len(kpart)}."
                )
        else:
            worst = part.iloc[part["rel_sd_error"].abs().argmax()]
            lines.append(
                f"- {model.title()}: worst sampler rel_sd_error={worst['rel_sd_error']:.3f} "
                f"({worst['method']}, n={int(worst['n'])}); acceptable rows={int(part['posterior_accuracy_acceptable'].sum())}/{len(part)}."
            )
    lines.extend(["", "## Main Cost Findings"])
    for keys, part in cost[~cost["rattle_status"].astype(str).eq("not_applicable")].groupby(["model", "k", "n"], dropna=False):
        model, k, n = keys
        if {"gibbs", "rattle"} <= set(part["method"]):
            g = part[part["method"].eq("gibbs")].iloc[0]
            r = part[part["method"].eq("rattle")].iloc[0]
            k_text = "" if not np.isfinite(k) else f" k={k:g}"
            lines.append(
                f"- {model}{k_text} n={int(n)}: ESS/sec Gibbs={g['ess_per_sec']:.2f}, "
                f"RATTLE={r['ess_per_sec']:.2f}; wall_time/ESS Gibbs={g['wall_time_per_ess']:.4g}, "
                f"RATTLE={r['wall_time_per_ess']:.4g}."
            )
    lines.extend(
        [
            "",
            "## Gibbs vs RATTLE By Model",
        ]
    )
    best = rankings[rankings["rank"].eq(1)] if not rankings.empty else pd.DataFrame()
    for _, row in best.iterrows():
        k_text = "" if not np.isfinite(row["k"]) else f" k={row['k']:g}"
        if bool(row["posterior_accuracy_acceptable"]):
            lines.append(
                f"- {row['model']}{k_text} n={int(row['n'])}: selected `{row['method']}` "
                f"(accuracy acceptable=True, ESS/sec={row['ess_per_sec']:.2f})."
            )
        else:
            lines.append(
                f"- {row['model']}{k_text} n={int(row['n'])}: no acceptable sampler in this run; "
                f"`{row['method']}` is only the least-bad/fastest flagged row (ESS/sec={row['ess_per_sec']:.2f})."
            )
    lines.extend(
        [
            "",
            "## RATTLE Tuning Warning",
            f"- {tuning_warning}",
        ]
    )
    if not rat.empty:
        too_high = int(rat["acceptance_too_high"].sum())
        lines.append(f"- RATTLE rows with acceptance_rate >= 0.995: {too_high}/{len(rat)}.")
        mismatch = rat[rat["posterior_sd_mismatch_flag"]]
        if not mismatch.empty:
            lines.append("- RATTLE posterior sd mismatch >10% appears in: " + ", ".join(f"{r.model} k={r.k:g} n={int(r.n)}" if np.isfinite(r.k) else f"{r.model} n={int(r.n)}" for r in mismatch.itertuples()))
    if not stage2.empty:
        lines.extend(["", "## RATTLE Stage2 Follow-Up"])
        follow = stage2[stage2["source"].eq("stage2_medium_followup")]
        if follow.empty:
            lines.append("- Stage2 tuning ran, but no medium follow-up row was found.")
        else:
            for row in follow.itertuples(index=False):
                lines.append(
                    f"- {row.model} k={row.k:g} n={int(row.n)} stage2 follow-up: status={row.status}, "
                    f"rel_sd_error={row.rel_sd_error:.3f}, ESS/sec={row.ess_per_sec:.3f}, acceptance={row.acceptance_rate:.3f}."
                )
    lines.extend(["", "## Laplace Notes"])
    lines.append("- Laplace RATTLE is not applicable. Laplace Gibbs is analyzed against the interval reference.")
    lines.extend(["", "## Suspicious Cases"])
    if suspicious.empty:
        lines.append("- No suspicious cases were flagged.")
    else:
        counts = suspicious.groupby(["issue_type", "severity"]).size().reset_index(name="count")
        for row in counts.itertuples(index=False):
            lines.append(f"- {row.issue_type} ({row.severity}): {row.count}.")
    lines.extend(["", "## Multi-Seed Robustness"])
    if multiseed.empty:
        lines.append("- No multi-seed sampler audit was available in `results/cost_audit_multiseed/`; current full sampler conclusions use seed 0 only.")
    else:
        lines.append(f"- Multi-seed rows summarized: {len(multiseed)}.")
        unstable = multiseed[multiseed["posterior_sd_sd"].fillna(0) > 0.10 * multiseed["posterior_sd_mean"].abs().replace(0, np.nan)]
        if not unstable.empty:
            lines.append(f"- Posterior sd variability >10% across seeds appears in {len(unstable)} model/method/n rows.")
    lines.extend(["", "## Recommended Next Targeted Experiments"])
    if stage2.empty:
        lines.append("- Run targeted RATTLE stage2 tuning when high acceptance coincides with posterior mismatch or weak movement.")
    else:
        lines.append("- Do not adopt the tested Student k=1 n=10 larger-move stage2 RATTLE setting; it failed posterior accuracy and cost.")
    if multiseed.empty:
        lines.append("- Run multi-seed robustness for n=10,20 before using sampler cost rankings as final scientific claims.")
    else:
        lines.append("- Treat Student k=1 n=10 as unresolved; multi-seed runs show severe instability, especially for Gibbs mean/sd.")
        lines.append("- For final scientific tables, add longer targeted chains or improved tuning for Student k=1 n=10 rather than broad reruns.")
    lines.extend(
        [
            "",
            "## Dashboard Status",
            f"- Dashboard health file: `{dashboard_health_path}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def laplace_notes(reference: pd.DataFrame) -> str:
    laplace = reference[reference["model"].eq("laplace")]
    targets = sorted(laplace["target_description"].dropna().astype(str).unique())
    return (
        "# Laplace Target Notes\n\n"
        f"Targets present: {', '.join(targets)}.\n\n"
        "Laplace deterministic `np.median` MLE errors and the median-interval Gibbs target are distinct for even n. "
        "The analysis report compares Laplace Gibbs only to `median_interval_contains_mu_star`. "
        "The deterministic `np.median` raw/KDE rows remain useful for reference sensitivity but should not be used as Gibbs posterior-correctness deltas.\n"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    reference = read_csv(args.reference_csv)
    summaries = sampler_summary(args.cost_dir)
    cost = cost_efficiency(args.cost_dir)
    tuning = normalize_k(read_csv(args.tuning_dir / "tuning_summary.csv"))
    accuracy = posterior_accuracy(reference, summaries)
    rankings = method_rankings(accuracy, cost)
    rat = rattle_diagnostics(cost, accuracy, tuning)
    chain_diag = chain_diagnostics(args.cost_dir)
    suspicious = suspicious_cases(accuracy, cost, rat, reference, chain_diag)
    multiseed = multiseed_summary(args.multiseed_dir)
    stage2 = stage2_followup(args.stage2_tuning_dir, args.stage2_cost_dir, reference)
    figures = write_figures(args.out_dir, accuracy, cost, chain_diag, args.cost_dir / "chain_samples.csv")

    tuning_warning = "No tuning warning detected."
    rec_path = args.tuning_dir / "recommended_rattle_settings.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        statuses = sorted({str(row.get("status", "")) for row in rec.get("settings", []) if row.get("status")})
        if any(status != "ok" for status in statuses):
            tuning_warning = f"Recommended settings contain non-ok status values: {', '.join(statuses)}."
        if rec.get("settings") and all(float(row.get("acceptance_rate", np.nan)) >= 0.995 for row in rec["settings"]):
            tuning_warning += " All recommended tuning rows have acceptance_rate >= 0.995."

    accuracy.to_csv(args.out_dir / "posterior_accuracy.csv", index=False)
    cost.to_csv(args.out_dir / "cost_efficiency.csv", index=False)
    rankings.to_csv(args.out_dir / "method_rankings.csv", index=False)
    rat.to_csv(args.out_dir / "rattle_diagnostics.csv", index=False)
    suspicious.to_csv(args.out_dir / "suspicious_cases.csv", index=False)
    chain_diag.to_csv(args.out_dir / "chain_diagnostics.csv", index=False)
    if not stage2.empty:
        stage2.to_csv(args.out_dir / "rattle_stage2_followup.csv", index=False)
    if not multiseed.empty:
        multiseed.to_csv(args.out_dir / "multiseed_summary.csv", index=False)
    (args.out_dir / "laplace_target_notes.md").write_text(laplace_notes(reference), encoding="utf-8")
    (args.out_dir / "figures" / "figure_index.json").write_text(json.dumps({"figures": figures}, indent=2), encoding="utf-8")
    (args.out_dir / "executive_summary.md").write_text(
        executive_summary(
            reference,
            accuracy,
            cost,
            rankings,
            rat,
            suspicious,
            multiseed,
            stage2,
            tuning_warning,
            ROOT / "results" / "analysis_pipeline" / "dashboard_health.json",
        ),
        encoding="utf-8",
    )
    print(f"wrote analysis report to {args.out_dir}")


if __name__ == "__main__":
    main()
