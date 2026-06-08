"""Summarize targeted validation outputs and recommend verdict updates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/targeted_validation_runs"))
    parser.add_argument("--reference-csv", type=Path, default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"))
    parser.add_argument("--verdict-csv", type=Path, default=Path("results/sampler_correctness_audit/final_sampler_verdict_table.csv"))
    return parser.parse_args()


def read_many(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"case_*/{name}")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frame["case_dir"] = str(path.parent)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def norm_k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" in out.columns:
        out["k"] = pd.to_numeric(out["k"], errors="coerce")
    return out


def reference_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    ref = norm_k(pd.read_csv(path))
    raw = ref[ref["estimator_type"].astype(str).isin(["raw_weighted_mc", "raw_mc_interval_reference"])].copy()
    return raw.rename(columns={"mean": "ref_mean", "sd": "ref_sd", "q025": "ref_q025", "q975": "ref_q975"})


def build_summary(runs_dir: Path, reference: pd.DataFrame) -> pd.DataFrame:
    posterior = norm_k(read_many(runs_dir, "posterior_summaries.csv"))
    if posterior.empty:
        return pd.DataFrame()
    key_cols = ["model", "k", "n"]
    merged = posterior.merge(reference[key_cols + ["ref_mean", "ref_sd", "target_description"]], on=key_cols, how="left", suffixes=("", "_reference"))
    merged["mean_abs_error_vs_reference"] = (pd.to_numeric(merged["mean_mu"], errors="coerce") - pd.to_numeric(merged["ref_mean"], errors="coerce")).abs()
    merged["sd_abs_error_vs_reference"] = (pd.to_numeric(merged["sd_mu"], errors="coerce") - pd.to_numeric(merged["ref_sd"], errors="coerce")).abs()
    return merged


def transition_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.groupby("case_id", dropna=False).agg(
        max_abs_constraint_residual=("abs_constraint_residual", "max"),
        mean_abs_constraint_residual=("abs_constraint_residual", "mean"),
        max_abs_pair_delta_error=("abs_pair_delta_error", "max"),
        mean_ESJD_mu=("ESJD_mu", "mean"),
        mean_movement_l2=("movement_l2", "mean"),
    ).reset_index()


def rattle_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [col for col in df.columns if col in {
        "case_id", "delta_H_mean_abs", "delta_H_rms", "delta_H_max_abs",
        "reverse_position_error", "reverse_momentum_error", "projection_failure_indicator",
        "reverse_check_failure_indicator", "tangent_residual_max", "ESJD_mu_mean", "movement_l2_mean",
    }]
    return df[cols].copy()


def branch_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="case_id", columns="branch_pair", values="frequency", fill_value=0.0).reset_index()


def initialization_summary(posterior: pd.DataFrame) -> pd.DataFrame:
    if posterior.empty or "initialization" not in posterior.columns:
        return pd.DataFrame()
    keys = ["model", "k", "n", "method", "seed"]
    grouped = posterior.groupby(keys, dropna=False).agg(
        initialization_count=("initialization", "nunique"),
        mean_mu_range=("mean_mu", lambda values: float(np.nanmax(values) - np.nanmin(values)) if len(values) else np.nan),
        sd_mu_range=("sd_mu", lambda values: float(np.nanmax(values) - np.nanmin(values)) if len(values) else np.nan),
    ).reset_index()
    return grouped


def recommendations(summary: pd.DataFrame, trans: pd.DataFrame, rattle: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary[["case_id", "model", "k", "n", "method", "seed", "initialization", "mean_abs_error_vs_reference", "sd_abs_error_vs_reference"]].copy()
    out = out.merge(trans[["case_id", "max_abs_constraint_residual", "max_abs_pair_delta_error"]] if not trans.empty else pd.DataFrame({"case_id": []}), on="case_id", how="left")
    out = out.merge(rattle[["case_id", "delta_H_mean_abs", "projection_failure_indicator", "reverse_check_failure_indicator"]] if not rattle.empty else pd.DataFrame({"case_id": []}), on="case_id", how="left")
    actions = []
    for _, row in out.iterrows():
        blockers = []
        if pd.notna(row.get("max_abs_constraint_residual")) and float(row["max_abs_constraint_residual"]) > 1e-5:
            blockers.append("constraint residual")
        if pd.notna(row.get("max_abs_pair_delta_error")) and float(row["max_abs_pair_delta_error"]) > 1e-4:
            blockers.append("pair delta")
        if pd.notna(row.get("projection_failure_indicator")) and int(row["projection_failure_indicator"]) != 0:
            blockers.append("projection failure")
        if pd.notna(row.get("reverse_check_failure_indicator")) and int(row["reverse_check_failure_indicator"]) != 0:
            blockers.append("reverse failure")
        if blockers:
            actions.append("requires sampler investigation")
        elif pd.notna(row.get("mean_abs_error_vs_reference")) and float(row["mean_abs_error_vs_reference"]) < 0.1:
            actions.append("upgrade caveat to clean")
        else:
            actions.append("keep caveat")
    out["recommended_action"] = actions
    return out


def report_text(summary: pd.DataFrame, recs: pd.DataFrame) -> str:
    lines = ["# Targeted Validation Report", ""]
    lines.append(f"- Completed posterior summary rows: {len(summary)}.")
    if not recs.empty:
        counts = recs["recommended_action"].value_counts().to_dict()
        lines.append(f"- Recommended action counts: {counts}.")
    lines.append("")
    lines.append("See CSV outputs in this directory for transition, RATTLE, branch, initialization, and blocker details.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_rows(args.reference_csv)
    summary = build_summary(args.runs_dir, reference)
    transition = transition_summary(read_many(args.runs_dir, "transition_diagnostics.csv"))
    rattle = rattle_summary(read_many(args.runs_dir, "rattle_energy_diagnostics.csv"))
    branch = branch_summary(read_many(args.runs_dir, "branch_diagnostics.csv"))
    init = initialization_summary(read_many(args.runs_dir, "posterior_summaries.csv"))
    recs = recommendations(summary, transition, rattle)
    blockers = recs[recs["recommended_action"].isin(["requires sampler investigation", "mark unresolved"])] if not recs.empty else pd.DataFrame()
    outputs = {
        "targeted_validation_summary.csv": summary,
        "transition_diagnostic_summary.csv": transition,
        "rattle_energy_summary.csv": rattle,
        "branch_usage_summary.csv": branch,
        "initialization_sensitivity_summary.csv": init,
        "upgraded_verdict_recommendations.csv": recs,
        "remaining_blockers.csv": blockers,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.runs_dir / name, index=False)
    (args.runs_dir / "targeted_validation_report.md").write_text(report_text(summary, recs), encoding="utf-8")
    print(f"wrote targeted validation analysis to {args.runs_dir}")


if __name__ == "__main__":
    main()
