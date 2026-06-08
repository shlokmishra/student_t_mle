"""Summarize targeted validation outputs and recommend verdict updates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["model", "k_key", "n", "method"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/targeted_validation_runs"))
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--verdict-csv",
        type=Path,
        default=Path("results/sampler_correctness_audit/final_sampler_verdict_table.csv"),
    )
    return parser.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_many(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"case_*/{name}")):
        frame = read_csv(path)
        if not frame.empty:
            frame["case_dir"] = str(path.parent)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def k_key(value) -> str:
    if pd.isna(value):
        return "NA"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def add_k_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" not in out.columns:
        out["k"] = np.nan
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["k_key"] = out["k"].map(k_key)
    return out


def reference_rows(path: Path) -> pd.DataFrame:
    ref = add_k_key(read_csv(path))
    if ref.empty:
        return pd.DataFrame()
    raw = ref[ref["estimator_type"].astype(str).eq("raw_weighted_mc")].copy()
    return (
        raw.groupby(["model", "k_key", "n"], dropna=False)
        .agg(ref_mean=("mean", "mean"), ref_sd=("sd", "mean"), ref_q025=("q025", "mean"), ref_q975=("q975", "mean"))
        .reset_index()
    )


def max_or_nan(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if len(values) else np.nan


def mean_or_nan(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def any_true(values: pd.Series) -> bool:
    if len(values) == 0:
        return False
    text = values.dropna().astype(str).str.lower()
    return bool(text.isin(["true", "1", "yes"]).any())


def build_summary(
    posterior: pd.DataFrame,
    transition: pd.DataFrame,
    rattle: pd.DataFrame,
    branch: pd.DataFrame,
    init: pd.DataFrame,
    reference: pd.DataFrame,
    verdicts: pd.DataFrame,
) -> pd.DataFrame:
    if posterior.empty:
        return pd.DataFrame()
    posterior = add_k_key(posterior)
    transition = add_k_key(transition) if not transition.empty else pd.DataFrame()
    rattle = add_k_key(rattle) if not rattle.empty else pd.DataFrame()
    branch = add_k_key(branch) if not branch.empty else pd.DataFrame()
    init = add_k_key(init) if not init.empty else pd.DataFrame()
    verdicts = add_k_key(verdicts) if not verdicts.empty else pd.DataFrame()

    summary = (
        posterior.groupby(KEYS, dropna=False)
        .agg(
            k=("k", "first"),
            num_runs=("case_id", "nunique"),
            num_seeds=("seed", "nunique"),
            num_initializations=("initialization", "nunique"),
            iterations_min=("iterations", "min"),
            iterations_max=("iterations", "max"),
            mean_mu_mean=("mean_mu", "mean"),
            mean_mu_sd=("mean_mu", "std"),
            mean_mu_min=("mean_mu", "min"),
            mean_mu_max=("mean_mu", "max"),
            sd_mu_mean=("sd_mu", "mean"),
            sd_mu_sd=("sd_mu", "std"),
            sd_mu_min=("sd_mu", "min"),
            sd_mu_max=("sd_mu", "max"),
            ess_mu_min=("ess_mu", "min"),
            ess_mu_median=("ess_mu", "median"),
            acceptance_rate_mean=("acceptance_rate", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(reference, on=["model", "k_key", "n"], how="left")
    summary["mean_error_over_ref_sd"] = (summary["mean_mu_mean"] - summary["ref_mean"]).abs() / summary["ref_sd"].replace(0, np.nan)
    summary["sd_rel_error"] = (summary["sd_mu_mean"] - summary["ref_sd"]).abs() / summary["ref_sd"].replace(0, np.nan)
    summary["mean_mu_range_over_ref_sd"] = (summary["mean_mu_max"] - summary["mean_mu_min"]).abs() / summary["ref_sd"].replace(0, np.nan)

    if not transition.empty:
        trans = (
            transition.groupby(KEYS, dropna=False)
            .agg(
                transition_rows=("case_id", "count"),
                max_abs_constraint_residual=("abs_constraint_residual", max_or_nan),
                mean_abs_constraint_residual=("abs_constraint_residual", mean_or_nan),
                max_abs_pair_delta_error=("abs_pair_delta_error", max_or_nan),
                mean_esjd_mu=("ESJD_mu", mean_or_nan),
                max_movement_l2=("movement_l2", max_or_nan),
                max_abs_delta_mu=("abs_delta_mu", max_or_nan),
                laplace_max_abs_median_residual=("median_minus_mu_star", lambda s: max_or_nan(s.abs()) if hasattr(s, "abs") else np.nan),
            )
            .reset_index()
        )
        summary = summary.merge(trans, on=KEYS, how="left")

    if not rattle.empty:
        rat = (
            rattle.groupby(KEYS, dropna=False)
            .agg(
                rattle_diag_rows=("case_id", "count"),
                rattle_diag_applicable=("diagnostic_applicable", any_true),
                max_delta_H_abs=("delta_H_max_abs", max_or_nan),
                mean_delta_H_abs=("delta_H_mean_abs", mean_or_nan),
                max_tangent_residual=("tangent_residual_max", max_or_nan),
                projection_failure_count=("projection_failure_count", max_or_nan),
                reverse_check_failure_count=("reverse_check_failure_count", max_or_nan),
                projection_failure_indicator=("projection_failure_indicator", max_or_nan),
                reverse_check_failure_indicator=("reverse_check_failure_indicator", max_or_nan),
                max_reverse_position_error=("max_reverse_position_error", max_or_nan),
                max_reverse_momentum_error=("max_reverse_momentum_error", max_or_nan),
            )
            .reset_index()
        )
        summary = summary.merge(rat, on=KEYS, how="left")

    if not branch.empty and "branch_switching_rate" in branch.columns:
        br = (
            branch.groupby(KEYS, dropna=False)
            .agg(
                branch_diag_rows=("case_id", "count"),
                branch_switching_rate_mean=("branch_switching_rate", mean_or_nan),
                branch_switching_rate_min=("branch_switching_rate", max_or_nan),
            )
            .reset_index()
        )
        summary = summary.merge(br, on=KEYS, how="left")

    if not init.empty:
        init_summary = (
            init.groupby(KEYS, dropna=False)
            .agg(
                max_initial_constraint_residual=("initial_constraint_residual", max_or_nan),
                max_initial_x_sd=("initial_x_sd", max_or_nan),
            )
            .reset_index()
        )
        summary = summary.merge(init_summary, on=KEYS, how="left")

    if not verdicts.empty:
        cols = KEYS + ["verdict", "safe_to_present", "main_warning", "main_reason"]
        cols = [col for col in cols if col in verdicts.columns]
        summary = summary.merge(verdicts[cols].drop_duplicates(KEYS), on=KEYS, how="left")
    return summary


def decision_for(row: pd.Series) -> tuple[str, str]:
    model = str(row["model"])
    method = str(row["method"])
    kk = str(row["k_key"])
    verdict = str(row.get("verdict", ""))
    safe = str(row.get("safe_to_present", ""))
    if model == "laplace" and method == "rattle":
        return "keep caveat", "Laplace RATTLE remains not applicable."

    mean_ok = pd.notna(row.get("mean_error_over_ref_sd")) and float(row["mean_error_over_ref_sd"]) <= 0.15
    sd_ok = pd.notna(row.get("sd_rel_error")) and float(row["sd_rel_error"]) <= 0.15
    init_ok = pd.isna(row.get("mean_mu_range_over_ref_sd")) or float(row["mean_mu_range_over_ref_sd"]) <= 0.25
    ess_ok = pd.notna(row.get("ess_mu_min")) and float(row["ess_mu_min"]) >= 100
    constraint_ok = pd.isna(row.get("max_abs_constraint_residual")) or float(row["max_abs_constraint_residual"]) <= 1e-6
    geometry_ok = True
    if method == "rattle":
        geometry_ok = (
            (pd.isna(row.get("projection_failure_indicator")) or float(row["projection_failure_indicator"]) == 0)
            and (pd.isna(row.get("reverse_check_failure_indicator")) or float(row["reverse_check_failure_indicator"]) == 0)
            and (pd.isna(row.get("max_delta_H_abs")) or float(row["max_delta_H_abs"]) <= 0.25)
            and (pd.isna(row.get("max_tangent_residual")) or float(row["max_tangent_residual"]) <= 1e-6)
        )

    if model == "student_t" and kk == "1" and int(row["n"]) == 10:
        if method == "gibbs":
            return "requires sampler investigation", "Targeted runs still have low ESS and large posterior error; heavy-tail local geometry remains unstable."
        if mean_ok and sd_ok and geometry_ok and ess_ok:
            return "keep caveat", "Targeted RATTLE improves k=1,n=10 but baseline target-sensitivity issue remains unresolved."
        return "mark unresolved", "Targeted k=1,n=10 evidence is improved but not enough for a clean upgrade."

    if not constraint_ok:
        return "requires sampler investigation", "Constraint residual exceeds tolerance in targeted diagnostics."
    if method == "rattle" and not geometry_ok:
        return "requires sampler investigation", "RATTLE projection/reverse/energy diagnostic exceeds tolerance."
    if not ess_ok:
        return "keep caveat", "Targeted posterior is plausible but minimum ESS remains low."
    if mean_ok and sd_ok and init_ok:
        return "upgrade to clean", "Targeted posterior, initialization, and geometry diagnostics pass cached thresholds."
    if verdict == "unresolved" or safe == "no":
        return "mark unresolved", "Targeted posterior agreement is still outside clean thresholds."
    return "keep caveat", "Targeted evidence improves diagnostics but not enough for clean threshold upgrade."


def recommendations(summary: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or verdicts.empty:
        return pd.DataFrame()
    verdicts = add_k_key(verdicts)
    caveats = verdicts[
        verdicts["safe_to_present"].astype(str).isin(["caveat_only", "no", "hide_or_mark_not_applicable"])
        | verdicts["verdict"].astype(str).isin(["unresolved", "not_applicable", "pass_with_warning"])
    ].copy()
    rows = []
    for _, base in caveats.iterrows():
        match = summary[
            summary["model"].eq(base["model"])
            & summary["k_key"].eq(base["k_key"])
            & summary["n"].eq(base["n"])
            & summary["method"].eq(base["method"])
        ]
        if match.empty:
            action = "keep caveat" if str(base["verdict"]) == "not_applicable" else "mark unresolved"
            if str(base["model"]) == "laplace" and str(base["method"]) == "rattle":
                reason = "Laplace RATTLE remains not applicable; targeted Laplace validation is Gibbs-only."
            else:
                reason = "No targeted validation runs found for this case."
            row = base.to_dict()
        else:
            row = match.iloc[0].to_dict()
            action, reason = decision_for(match.iloc[0])
        row.update(
            {
                "old_verdict": base.get("verdict", ""),
                "old_safe_to_present": base.get("safe_to_present", ""),
                "recommended_action": action,
                "recommendation_reason": reason,
                "new_recommended_verdict": {
                    "upgrade to clean": "pass",
                    "keep caveat": "pass_with_warning" if str(base.get("verdict")) != "not_applicable" else "not_applicable",
                    "mark unresolved": "unresolved",
                    "requires sampler investigation": "unresolved",
                }[action],
                "new_safe_to_present": {
                    "upgrade to clean": "yes",
                    "keep caveat": "caveat_only" if str(base.get("verdict")) != "not_applicable" else "hide_or_mark_not_applicable",
                    "mark unresolved": "no",
                    "requires sampler investigation": "no",
                }[action],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def remaining_blockers(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return pd.DataFrame()
    return recs[recs["recommended_action"].isin(["mark unresolved", "requires sampler investigation"])].copy()


def report_text(summary: pd.DataFrame, recs: pd.DataFrame, blockers: pd.DataFrame) -> str:
    lines = ["# Targeted Validation Report", ""]
    lines.append(f"- Grouped targeted validation rows: {len(summary)}.")
    if not recs.empty:
        lines.append(f"- Recommendation counts: {recs['recommended_action'].value_counts().to_dict()}.")
    if not blockers.empty:
        lines.append(f"- Remaining blockers: {len(blockers)}.")
    lines.extend(["", "Targeted validation decisions are conditional on raw weighted-MC references and cached diagnostic tolerances."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_rows(args.reference_csv)
    verdicts = read_csv(args.verdict_csv)
    posterior = read_many(args.runs_dir, "posterior_summaries.csv")
    transition = read_many(args.runs_dir, "transition_diagnostics.csv")
    rattle = read_many(args.runs_dir, "rattle_energy_diagnostics.csv")
    branch = read_many(args.runs_dir, "branch_diagnostics.csv")
    init = read_many(args.runs_dir, "initialization_diagnostics.csv")
    summary = build_summary(posterior, transition, rattle, branch, init, reference, verdicts)
    recs = recommendations(summary, verdicts)
    blockers = remaining_blockers(recs)
    transition_summary = (
        add_k_key(transition)
        .groupby(KEYS, dropna=False)
        .agg(
            rows=("case_id", "count"),
            max_abs_constraint_residual=("abs_constraint_residual", max_or_nan),
            max_abs_pair_delta_error=("abs_pair_delta_error", max_or_nan),
            mean_esjd_mu=("ESJD_mu", mean_or_nan),
            max_movement_l2=("movement_l2", max_or_nan),
        )
        .reset_index()
        if not transition.empty
        else pd.DataFrame()
    )
    rattle_summary = (
        add_k_key(rattle)
        .groupby(KEYS, dropna=False)
        .agg(
            rows=("case_id", "count"),
            max_delta_H_abs=("delta_H_max_abs", max_or_nan),
            mean_delta_H_abs=("delta_H_mean_abs", mean_or_nan),
            max_tangent_residual=("tangent_residual_max", max_or_nan),
            projection_failure_indicator=("projection_failure_indicator", max_or_nan),
            reverse_check_failure_indicator=("reverse_check_failure_indicator", max_or_nan),
        )
        .reset_index()
        if not rattle.empty
        else pd.DataFrame()
    )
    branch_usage = (
        add_k_key(branch).pivot_table(index=KEYS, columns="branch_pair", values="frequency", aggfunc="mean", fill_value=0).reset_index()
        if not branch.empty and "branch_pair" in branch.columns
        else pd.DataFrame()
    )
    init_summary = (
        add_k_key(posterior)
        .groupby(KEYS, dropna=False)
        .agg(
            num_initializations=("initialization", "nunique"),
            mean_mu_range=("mean_mu", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
            sd_mu_range=("sd_mu", lambda s: float(pd.to_numeric(s, errors="coerce").max() - pd.to_numeric(s, errors="coerce").min())),
        )
        .reset_index()
        if not posterior.empty
        else pd.DataFrame()
    )
    outputs = {
        "targeted_validation_summary.csv": summary,
        "transition_diagnostic_summary.csv": transition_summary,
        "rattle_energy_summary.csv": rattle_summary,
        "branch_usage_summary.csv": branch_usage,
        "initialization_sensitivity_summary.csv": init_summary,
        "upgraded_verdict_recommendations.csv": recs,
        "remaining_blockers.csv": blockers,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.runs_dir / name, index=False)
    (args.runs_dir / "targeted_validation_report.md").write_text(report_text(summary, recs, blockers), encoding="utf-8")
    print(f"wrote targeted validation analysis to {args.runs_dir}")


if __name__ == "__main__":
    main()
