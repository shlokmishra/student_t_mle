"""Reconcile production baseline runs with targeted validation diagnostics.

Targeted validation adds diagnostic and initialization evidence, but should not
replace production-length posterior or efficiency estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["model", "k_key", "n", "method"]
POSTERIOR_COLS = ["mean", "sd", "q025", "q50", "q975"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/cost_audit"))
    parser.add_argument("--targeted-dir", type=Path, default=Path("results/targeted_validation_runs"))
    parser.add_argument(
        "--targeted-final-verdict",
        type=Path,
        default=Path("results/refreshed_analysis/targeted_validation/correctness/final_sampler_verdict_table.csv"),
    )
    parser.add_argument(
        "--targeted-recommendations",
        type=Path,
        default=Path("results/targeted_validation_runs/upgraded_verdict_recommendations.csv"),
    )
    parser.add_argument(
        "--baseline-final-verdict",
        type=Path,
        default=Path("results/sampler_correctness_audit/final_sampler_verdict_table.csv"),
    )
    parser.add_argument(
        "--baseline-coverage",
        type=Path,
        default=Path("results/sampler_correctness_audit/diagnostic_coverage_table.csv"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/refreshed_analysis/targeted_validation"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


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


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if len(values) else np.nan


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else np.nan


def mean_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def read_many(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"case_*/{name}")):
        frame = read_csv(path)
        if not frame.empty:
            frame["case_dir"] = str(path.parent)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def reference_summary(path: Path) -> pd.DataFrame:
    ref = add_k_key(read_csv(path))
    if ref.empty:
        return pd.DataFrame(columns=["model", "k_key", "n"])
    raw = ref[ref["estimator_type"].astype(str).eq("raw_weighted_mc")].copy()
    return (
        raw.groupby(["model", "k_key", "n"], dropna=False)
        .agg(
            raw_weighted_mc_mean=("mean", "mean"),
            raw_weighted_mc_sd=("sd", "mean"),
            raw_weighted_mc_q025=("q025", "mean"),
            raw_weighted_mc_q50=("q50", "mean"),
            raw_weighted_mc_q975=("q975", "mean"),
            raw_weighted_mc_B=("B", "max"),
            raw_weighted_mc_weighted_ess=("weighted_ess", "mean"),
        )
        .reset_index()
    )


def aggregate_runs(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEYS)
    df = add_k_key(df)
    if "iterations" not in df.columns and "num_iterations" in df.columns:
        df["iterations"] = df["num_iterations"]
    grouped = (
        df.groupby(KEYS, dropna=False)
        .agg(
            k=("k", "first"),
            **{
                f"{prefix}_num_runs": ("method", "count"),
                f"{prefix}_num_seeds": ("seed", "nunique"),
                f"{prefix}_num_iterations_min": ("num_iterations", min_or_nan),
                f"{prefix}_num_iterations_max": ("num_iterations", max_or_nan),
                f"{prefix}_burn_in_min": ("burn_in", min_or_nan),
                f"{prefix}_burn_in_max": ("burn_in", max_or_nan),
                f"{prefix}_iterations_min": ("iterations", min_or_nan),
                f"{prefix}_iterations_max": ("iterations", max_or_nan),
                f"{prefix}_mean": ("mean_mu", mean_or_nan),
                f"{prefix}_sd": ("sd_mu", mean_or_nan),
                f"{prefix}_q025": ("q025_mu", mean_or_nan),
                f"{prefix}_q50": ("q50_mu", mean_or_nan),
                f"{prefix}_q975": ("q975_mu", mean_or_nan),
                f"{prefix}_ess_mu_min": ("ess_mu", min_or_nan),
                f"{prefix}_ess_mu_median": ("ess_mu", median_or_nan),
                f"{prefix}_ess_per_sec_median": ("ess_per_sec", median_or_nan),
                f"{prefix}_ess_per_sec_mean": ("ess_per_sec", mean_or_nan),
            },
        )
        .reset_index()
    )
    if prefix == "targeted" and "initialization" in df.columns:
        init = (
            df.groupby(KEYS, dropna=False)
            .agg(
                targeted_num_initializations=("initialization", "nunique"),
                targeted_mean_range=("mean_mu", lambda s: max_or_nan(s) - min_or_nan(s)),
                targeted_sd_range=("sd_mu", lambda s: max_or_nan(s) - min_or_nan(s)),
            )
            .reset_index()
        )
        grouped = grouped.merge(init, on=KEYS, how="left")
    return grouped


def aggregate_cost(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEYS)
    df = add_k_key(df)
    return (
        df.groupby(KEYS, dropna=False)
        .agg(
            **{
                f"{prefix}_cost_num_iterations_min": ("num_iterations", min_or_nan),
                f"{prefix}_cost_burn_in_min": ("burn_in", min_or_nan),
                f"{prefix}_cost_ess_mu_min": ("ess_mu", min_or_nan),
                f"{prefix}_cost_ess_per_sec_median": ("ess_per_sec", median_or_nan),
                f"{prefix}_cost_ess_per_sec_mean": ("ess_per_sec", mean_or_nan),
                f"{prefix}_cost_wall_time_median": ("wall_time_sec", median_or_nan),
            }
        )
        .reset_index()
    )


def aggregate_targeted_diagnostics(root: Path) -> pd.DataFrame:
    posterior = read_many(root, "posterior_summaries.csv")
    base = aggregate_runs(posterior, "targeted")

    transition = add_k_key(read_many(root, "transition_diagnostics.csv"))
    if not transition.empty:
        trans = (
            transition.groupby(KEYS, dropna=False)
            .agg(
                targeted_transition_rows=("case_id", "count"),
                targeted_max_abs_constraint_residual=("abs_constraint_residual", max_or_nan),
                targeted_max_abs_pair_delta_error=("abs_pair_delta_error", max_or_nan),
            )
            .reset_index()
        )
        base = base.merge(trans, on=KEYS, how="left")

    branch = add_k_key(read_many(root, "branch_diagnostics.csv"))
    if not branch.empty:
        branch_summary = (
            branch.groupby(KEYS, dropna=False)
            .agg(
                targeted_branch_rows=("case_id", "count"),
                targeted_branch_switching_rate_mean=("branch_switching_rate", mean_or_nan),
                targeted_branch_switching_rate_min=("branch_switching_rate", min_or_nan),
            )
            .reset_index()
        )
        base = base.merge(branch_summary, on=KEYS, how="left")

    rattle = add_k_key(read_many(root, "rattle_energy_diagnostics.csv"))
    if not rattle.empty:
        rattle_summary = (
            rattle.groupby(KEYS, dropna=False)
            .agg(
                targeted_rattle_energy_rows=("case_id", "count"),
                targeted_max_delta_H_abs=("delta_H_max_abs", max_or_nan),
                targeted_max_tangent_residual=("tangent_residual_max", max_or_nan),
                targeted_projection_failure_indicator=("projection_failure_indicator", max_or_nan),
                targeted_reverse_check_failure_indicator=("reverse_check_failure_indicator", max_or_nan),
            )
            .reset_index()
        )
        base = base.merge(rattle_summary, on=KEYS, how="left")

    init = add_k_key(read_many(root, "initialization_diagnostics.csv"))
    if not init.empty:
        init_summary = (
            init.groupby(KEYS, dropna=False)
            .agg(
                targeted_initialization_rows=("case_id", "count"),
                targeted_max_initial_constraint_residual=("initial_constraint_residual", max_or_nan),
                targeted_max_initial_x_sd=("initial_x_sd", max_or_nan),
            )
            .reset_index()
        )
        base = base.merge(init_summary, on=KEYS, how="left")

    return base


def smoke_contamination(root: Path) -> pd.DataFrame:
    records = []
    for meta_path in sorted(root.glob("case_*/run_metadata.json")):
        data = json.loads(meta_path.read_text())
        case = data.get("case") or {}
        ps_path = meta_path.parent / "posterior_summaries.csv"
        num_iterations = data.get("num_iterations", case.get("num_iterations"))
        burn_in = data.get("burn_in", case.get("burn_in"))
        run_type = str(data.get("run_type") or data.get("run_status") or "").lower()
        if ps_path.exists():
            ps = read_csv(ps_path)
            if not ps.empty:
                num_iterations = ps.iloc[0].get("num_iterations", num_iterations)
                burn_in = ps.iloc[0].get("burn_in", burn_in)
                run_type = str(ps.iloc[0].get("run_type", ps.iloc[0].get("run_status", run_type)) or run_type).lower()
        num_iterations = pd.to_numeric(pd.Series([num_iterations]), errors="coerce").iloc[0]
        burn_in = pd.to_numeric(pd.Series([burn_in]), errors="coerce").iloc[0]
        if (pd.notna(num_iterations) and num_iterations < 10000) or (pd.notna(burn_in) and burn_in < 1000) or run_type == "smoke":
            records.append(
                {
                    "case_id": case.get("case_id") or meta_path.parent.name.removeprefix("case_"),
                    "model": case.get("model"),
                    "k": case.get("k"),
                    "n": case.get("n"),
                    "method": case.get("method"),
                    "seed": case.get("seed"),
                    "initialization": case.get("initialization"),
                    "num_iterations": int(num_iterations) if pd.notna(num_iterations) else np.nan,
                    "burn_in": int(burn_in) if pd.notna(burn_in) else np.nan,
                    "output_dir": str(meta_path.parent),
                }
            )
    return pd.DataFrame(
        records,
        columns=["case_id", "model", "k", "n", "method", "seed", "initialization", "num_iterations", "burn_in", "output_dir"],
    )


def baseline_diagnostic_flags(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame(columns=KEYS)
    coverage = add_k_key(coverage)
    diag_cols = [
        "pair_delta_preservation",
        "branch_usage",
        "chain_split_stability",
        "ESS_autocorrelation",
        "multiseed_stability",
        "RATTLE_Hamiltonian_error",
        "RATTLE_tangent_residual",
    ]
    keep = KEYS + [c for c in diag_cols if c in coverage.columns]
    out = coverage[keep].copy()

    missing_cols = [c for c in diag_cols if c in out.columns]

    def missing(row: pd.Series) -> str:
        vals = []
        for col in missing_cols:
            value = str(row.get(col, "")).lower()
            if value in {"no", "missing", "not_available", "approximate"}:
                vals.append(col)
        return ";".join(vals)

    out["diagnostics_missing_in_baseline"] = out.apply(missing, axis=1)
    return out


def targeted_diag_status(row: pd.Series) -> dict[str, str]:
    method = str(row.get("method", ""))
    pair_delta = "yes" if pd.notna(row.get("targeted_max_abs_pair_delta_error")) else "no"
    branch = "yes" if pd.to_numeric(pd.Series([row.get("targeted_branch_rows")]), errors="coerce").fillna(0).iloc[0] > 0 else "no"
    init = "yes" if pd.to_numeric(pd.Series([row.get("targeted_num_initializations")]), errors="coerce").fillna(0).iloc[0] > 1 else "no"
    multiseed = "yes" if pd.to_numeric(pd.Series([row.get("targeted_num_seeds")]), errors="coerce").fillna(0).iloc[0] > 1 else "no"
    delta_h = "yes" if pd.notna(row.get("targeted_max_delta_H_abs")) else ("not_applicable" if method != "rattle" else "no")
    tangent = "yes" if pd.notna(row.get("targeted_max_tangent_residual")) else ("not_applicable" if method != "rattle" else "no")
    return {
        "targeted_pair_delta_available": pair_delta,
        "targeted_branch_switching_available": branch,
        "targeted_rattle_delta_H_available": delta_h,
        "targeted_rattle_tangent_residual_available": tangent,
        "targeted_initialization_sensitivity_available": init,
        "targeted_multiseed_stability_available": multiseed,
    }


def posterior_status(row: pd.Series, prefix: str) -> str:
    raw_sd = row.get("raw_weighted_mc_sd")
    if pd.isna(raw_sd) or raw_sd == 0 or pd.isna(row.get(f"{prefix}_mean")):
        return "missing"
    mean_err = abs(row[f"{prefix}_mean"] - row["raw_weighted_mc_mean"]) / raw_sd
    sd_rel = abs(row[f"{prefix}_sd"] - row["raw_weighted_mc_sd"]) / raw_sd
    q025_err = abs(row[f"{prefix}_q025"] - row["raw_weighted_mc_q025"]) / raw_sd
    q975_err = abs(row[f"{prefix}_q975"] - row["raw_weighted_mc_q975"]) / raw_sd
    ess_min = row.get(f"{prefix}_ess_mu_min")
    if mean_err <= 0.15 and sd_rel <= 0.25 and q025_err <= 0.35 and q975_err <= 0.35 and pd.notna(ess_min) and ess_min >= 100:
        return "good"
    if mean_err <= 0.35 and sd_rel <= 0.50 and q025_err <= 0.75 and q975_err <= 0.75:
        return "warning"
    return "failing"


def diagnostics_good(row: pd.Series) -> bool:
    if str(row.get("model")) == "laplace" and str(row.get("method")) == "rattle":
        return False
    if row.get("targeted_smoke_contaminated", False):
        return False
    constraint = row.get("targeted_max_abs_constraint_residual")
    if pd.notna(constraint) and float(constraint) > 1e-6:
        return False
    if str(row.get("method")) == "rattle":
        for col in ["targeted_projection_failure_indicator", "targeted_reverse_check_failure_indicator"]:
            value = row.get(col)
            if pd.notna(value) and float(value) != 0.0:
                return False
        delta_h = row.get("targeted_max_delta_H_abs")
        if pd.notna(delta_h) and float(delta_h) > 0.25:
            return False
        tangent = row.get("targeted_max_tangent_residual")
        if pd.notna(tangent) and float(tangent) > 1e-6:
            return False
    if pd.to_numeric(pd.Series([row.get("targeted_num_initializations")]), errors="coerce").fillna(0).iloc[0] < 1:
        return False
    return True


def material_disagreement(row: pd.Series) -> bool:
    raw_sd = row.get("raw_weighted_mc_sd")
    if pd.isna(raw_sd) or raw_sd == 0:
        return False
    mean_diff = abs(row.get("baseline_mean", np.nan) - row.get("targeted_mean", np.nan)) / raw_sd
    sd_diff = abs(row.get("baseline_sd", np.nan) - row.get("targeted_sd", np.nan)) / raw_sd
    return (pd.notna(mean_diff) and mean_diff > 0.35) or (pd.notna(sd_diff) and sd_diff > 0.35)


def decide(row: pd.Series) -> tuple[str, str, bool, str, str]:
    model = str(row.get("model"))
    method = str(row.get("method"))
    k = str(row.get("k_key"))
    n = int(row.get("n"))
    targeted_action = str(row.get("targeted_recommendation", ""))
    base_status = str(row.get("baseline_posterior_status"))
    targ_status = str(row.get("targeted_posterior_status"))
    diag_good = bool(row.get("targeted_diagnostics_good"))
    conflict = bool(row.get("baseline_targeted_material_disagreement"))

    if model == "laplace" and method == "rattle":
        return "not_applicable", "baseline_100k", False, "Laplace RATTLE is not applicable.", "not_applicable"

    if model == "student_t" and k == "1" and n == 10:
        return "diagnostic_unresolved", "conflict", True, "Student k=1,n=10 remains a separate heavy-tail diagnostic case.", "diagnostic_unresolved"

    if conflict:
        return (
            "unresolved_until_production_rerun",
            "conflict",
            True,
            "Baseline 100k and targeted 30k posterior summaries disagree materially.",
            "conflict",
        )

    if base_status == "good" and diag_good and targeted_action == "upgrade to clean":
        return "clean", "baseline_100k", False, "Baseline 100k posterior is good; targeted validation fills diagnostics.", "targeted_validation"

    if base_status in {"warning", "failing", "missing"} and targ_status == "good" and diag_good and targeted_action == "upgrade to clean":
        return (
            "unresolved_until_production_rerun",
            "conflict",
            True,
            "Targeted 30k is good but production-length baseline posterior is not clean enough to replace.",
            "targeted_validation",
        )

    if targeted_action == "keep caveat":
        if base_status == "good" and diag_good:
            return "clean", "baseline_100k", False, "Only missing diagnostics were at issue; targeted validation filled them.", "targeted_validation"
        return "caveat", "baseline_100k", False, "Targeted validation did not justify a clean upgrade.", "targeted_validation"

    if targeted_action in {"mark unresolved", "requires sampler investigation"}:
        return "diagnostic_unresolved", "conflict", True, "Targeted validation remains unresolved or requests sampler investigation.", "targeted_validation"

    if base_status == "good" and diag_good:
        return "clean", "baseline_100k", False, "Baseline posterior and targeted diagnostics are acceptable.", "targeted_validation"

    return "caveat", "baseline_100k", False, "No production rerun required by reconciliation, but evidence is not a clean upgrade.", "targeted_validation"


def build_reconciliation(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = aggregate_runs(read_csv(args.baseline_dir / "posterior_summaries.csv"), "baseline")
    cost = aggregate_cost(read_csv(args.baseline_dir / "cost_ledger.csv"), "baseline")
    if not cost.empty:
        cost_cols = KEYS + ["baseline_cost_ess_per_sec_median", "baseline_cost_ess_per_sec_mean"]
        baseline = baseline.merge(cost[cost_cols], on=KEYS, how="left")
        baseline["baseline_ess_per_sec_median"] = baseline["baseline_cost_ess_per_sec_median"].combine_first(
            baseline["baseline_ess_per_sec_median"]
        )
        baseline["baseline_ess_per_sec_mean"] = baseline["baseline_cost_ess_per_sec_mean"].combine_first(baseline["baseline_ess_per_sec_mean"])
        baseline = baseline.drop(columns=[c for c in ["baseline_cost_ess_per_sec_median", "baseline_cost_ess_per_sec_mean"] if c in baseline])

    targeted = aggregate_targeted_diagnostics(args.targeted_dir)
    ref = reference_summary(args.reference_csv)
    old = add_k_key(read_csv(args.baseline_final_verdict))
    targeted_final = add_k_key(read_csv(args.targeted_final_verdict))
    recs = add_k_key(read_csv(args.targeted_recommendations))
    coverage = baseline_diagnostic_flags(read_csv(args.baseline_coverage))
    smoke = smoke_contamination(args.targeted_dir)

    include = pd.concat(
        [
            recs[KEYS] if not recs.empty else pd.DataFrame(columns=KEYS),
            old[old.get("safe_to_present", "").astype(str).isin(["caveat_only", "no", "hide_or_mark_not_applicable"])][KEYS]
            if not old.empty
            else pd.DataFrame(columns=KEYS),
        ],
        ignore_index=True,
    ).drop_duplicates()

    out = include.merge(baseline, on=KEYS, how="left").merge(targeted, on=KEYS, how="left", suffixes=("", "_targeteddup"))
    if "k_targeteddup" in out.columns:
        out["k"] = out["k"].combine_first(out["k_targeteddup"])
        out = out.drop(columns=["k_targeteddup"])
    out = out.merge(ref, on=["model", "k_key", "n"], how="left")

    old_cols = KEYS + ["verdict", "safe_to_present", "main_warning", "main_reason"]
    old_cols = [c for c in old_cols if c in old.columns]
    out = out.merge(
        old[old_cols].rename(
            columns={
                "verdict": "old_baseline_verdict",
                "safe_to_present": "old_baseline_safe_to_present",
                "main_warning": "old_baseline_warning",
                "main_reason": "old_baseline_reason",
            }
        ),
        on=KEYS,
        how="left",
    )

    rec_cols = KEYS + ["recommended_action", "recommendation_reason", "new_recommended_verdict", "new_safe_to_present"]
    rec_cols = [c for c in rec_cols if c in recs.columns]
    out = out.merge(
        recs[rec_cols].rename(
            columns={
                "recommended_action": "targeted_recommendation",
                "recommendation_reason": "targeted_recommendation_reason",
                "new_recommended_verdict": "targeted_new_recommended_verdict",
                "new_safe_to_present": "targeted_new_safe_to_present",
            }
        ),
        on=KEYS,
        how="left",
    )

    tf_cols = KEYS + ["verdict", "safe_to_present"]
    tf_cols = [c for c in tf_cols if c in targeted_final.columns]
    out = out.merge(
        targeted_final[tf_cols].rename(columns={"verdict": "targeted_final_verdict", "safe_to_present": "targeted_final_safe_to_present"}),
        on=KEYS,
        how="left",
    )
    out = out.merge(coverage[KEYS + ["diagnostics_missing_in_baseline"]], on=KEYS, how="left")

    smoke_keys = set()
    if not smoke.empty:
        smoke = add_k_key(smoke)
        smoke_keys = set(map(tuple, smoke[KEYS].drop_duplicates().itertuples(index=False, name=None)))
    out["targeted_smoke_contaminated"] = out[KEYS].apply(lambda r: tuple(r) in smoke_keys, axis=1)

    out["targeted_is_shorter"] = out["targeted_num_iterations_min"] < out["baseline_num_iterations_min"]
    for prefix in ["baseline", "targeted"]:
        out[f"{prefix}_error_mean_over_raw_sd"] = (out[f"{prefix}_mean"] - out["raw_weighted_mc_mean"]).abs() / out[
            "raw_weighted_mc_sd"
        ].replace(0, np.nan)
        out[f"{prefix}_error_sd_over_raw_sd"] = (out[f"{prefix}_sd"] - out["raw_weighted_mc_sd"]).abs() / out[
            "raw_weighted_mc_sd"
        ].replace(0, np.nan)
        out[f"{prefix}_error_q025_over_raw_sd"] = (out[f"{prefix}_q025"] - out["raw_weighted_mc_q025"]).abs() / out[
            "raw_weighted_mc_sd"
        ].replace(0, np.nan)
        out[f"{prefix}_error_q50_over_raw_sd"] = (out[f"{prefix}_q50"] - out["raw_weighted_mc_q50"]).abs() / out[
            "raw_weighted_mc_sd"
        ].replace(0, np.nan)
        out[f"{prefix}_error_q975_over_raw_sd"] = (out[f"{prefix}_q975"] - out["raw_weighted_mc_q975"]).abs() / out[
            "raw_weighted_mc_sd"
        ].replace(0, np.nan)
        out[f"{prefix}_posterior_status"] = out.apply(lambda row: posterior_status(row, prefix), axis=1)

    out["baseline_targeted_mean_diff_over_raw_sd"] = (out["baseline_mean"] - out["targeted_mean"]).abs() / out[
        "raw_weighted_mc_sd"
    ].replace(0, np.nan)
    out["baseline_targeted_sd_diff_over_raw_sd"] = (out["baseline_sd"] - out["targeted_sd"]).abs() / out[
        "raw_weighted_mc_sd"
    ].replace(0, np.nan)
    out["baseline_targeted_material_disagreement"] = out.apply(material_disagreement, axis=1)

    diag_status = out.apply(lambda row: pd.Series(targeted_diag_status(row)), axis=1)
    out = pd.concat([out, diag_status], axis=1)
    out["targeted_diagnostics_good"] = out.apply(diagnostics_good, axis=1)

    decisions = out.apply(lambda row: pd.Series(decide(row)), axis=1)
    decisions.columns = [
        "reconciled_verdict",
        "evidence_source_for_posterior",
        "production_rerun_needed",
        "reason",
        "evidence_source_for_diagnostics",
    ]
    out = pd.concat([out, decisions], axis=1)
    out["use_baseline_for_posterior_efficiency"] = True
    out["targeted_efficiency_is_diagnostic_only"] = True

    ordered = [
        "model",
        "k",
        "k_key",
        "n",
        "method",
        "baseline_num_iterations_min",
        "baseline_num_iterations_max",
        "baseline_burn_in_min",
        "baseline_burn_in_max",
        "targeted_num_iterations_min",
        "targeted_num_iterations_max",
        "targeted_burn_in_min",
        "targeted_burn_in_max",
        "targeted_is_shorter",
        "baseline_mean",
        "baseline_sd",
        "baseline_q025",
        "baseline_q50",
        "baseline_q975",
        "targeted_mean",
        "targeted_sd",
        "targeted_q025",
        "targeted_q50",
        "targeted_q975",
        "raw_weighted_mc_mean",
        "raw_weighted_mc_sd",
        "raw_weighted_mc_q025",
        "raw_weighted_mc_q50",
        "raw_weighted_mc_q975",
        "baseline_error_mean_over_raw_sd",
        "baseline_error_sd_over_raw_sd",
        "targeted_error_mean_over_raw_sd",
        "targeted_error_sd_over_raw_sd",
        "baseline_targeted_mean_diff_over_raw_sd",
        "baseline_targeted_sd_diff_over_raw_sd",
        "baseline_ess_per_sec_median",
        "targeted_ess_per_sec_median",
        "old_baseline_verdict",
        "targeted_recommendation",
        "reconciled_verdict",
        "evidence_source_for_posterior",
        "evidence_source_for_diagnostics",
        "production_rerun_needed",
        "reason",
        "diagnostics_missing_in_baseline",
        "targeted_pair_delta_available",
        "targeted_branch_switching_available",
        "targeted_rattle_delta_H_available",
        "targeted_rattle_tangent_residual_available",
        "targeted_initialization_sensitivity_available",
        "targeted_multiseed_stability_available",
        "targeted_diagnostics_good",
        "targeted_smoke_contaminated",
        "use_baseline_for_posterior_efficiency",
        "targeted_efficiency_is_diagnostic_only",
    ]
    remaining = [c for c in out.columns if c not in ordered]
    return out[[c for c in ordered if c in out.columns] + remaining], smoke


def winner_changes(out_dir: Path) -> tuple[str, pd.DataFrame]:
    baseline_path = Path("results/efficiency_audit/method_winners.csv")
    targeted_path = out_dir / "efficiency" / "method_winners.csv"
    if not baseline_path.exists() or not targeted_path.exists():
        return "Efficiency winner comparison unavailable.", pd.DataFrame()
    base = add_k_key(read_csv(baseline_path))
    target = add_k_key(read_csv(targeted_path))
    cols = ["model", "k_key", "n", "recommended_efficiency_winner"]
    merged = base[cols].merge(target[cols], on=["model", "k_key", "n"], how="inner", suffixes=("_baseline", "_targeted"))
    changed = merged[
        merged["recommended_efficiency_winner_baseline"].fillna("missing")
        != merged["recommended_efficiency_winner_targeted"].fillna("missing")
    ].copy()
    if changed.empty:
        return "No final dashboard efficiency winners should change; use baseline production winners.", changed
    return (
        "Some overlapping targeted diagnostic winners differ from baseline, but final dashboard winners should remain baseline production winners.",
        changed,
    )


def write_report(out: pd.DataFrame, smoke: pd.DataFrame, out_dir: Path) -> None:
    clean = out[out["reconciled_verdict"].eq("clean")].copy()
    rerun = out[out["production_rerun_needed"].astype(bool)].copy()
    k2 = out[(out["model"].eq("student_t")) & (out["k_key"].eq("2")) & (out["n"].eq(10))]
    winner_line, winner_diff = winner_changes(out_dir)

    def bullets(df: pd.DataFrame, cols: list[str]) -> list[str]:
        if df.empty:
            return ["- None."]
        lines = []
        sort_cols = [c for c in ["model", "k_key", "n", "method"] if c in df.columns]
        for _, row in df.sort_values(sort_cols).iterrows():
            method = f" {row['method']}" if "method" in row.index and pd.notna(row.get("method")) else ""
            label = f"{row['model']} k={row['k_key']} n={int(row['n'])}{method}"
            details = "; ".join(f"{col}={row[col]}" for col in cols if col in row and pd.notna(row[col]))
            lines.append(f"- {label}: {details}")
        return lines

    lines = [
        "# Baseline vs Targeted Validation Reconciliation",
        "",
        "Targeted validation is treated as diagnostic evidence only. Production posterior and efficiency estimates stay anchored to the 100k/20k baseline runs.",
        "",
        "## Truly clean without more runs",
        *bullets(clean, ["reason"]),
        "",
        "## Production reruns needed",
        *bullets(rerun, ["reconciled_verdict", "reason"]),
        "",
        "## k=2,n=10 status",
    ]
    if not k2.empty:
        for _, row in k2.sort_values("method").iterrows():
            lines.append(
                f"- {row['method']}: {row['reconciled_verdict']}; smoke_contaminated={row['targeted_smoke_contaminated']}; "
                f"targeted iterations={int(row['targeted_num_iterations_min'])}-{int(row['targeted_num_iterations_max'])}; "
                f"reason={row['reason']}"
            )
    else:
        lines.append("- Not present in reconciliation table.")
    lines.extend(
        [
            "",
            "## Dashboard numbers",
            "- Use baseline 100k/20k runs for posterior summaries and efficiency.",
            "- Use targeted validation for diagnostics, initialization sensitivity, and multiseed/branch/geometric evidence.",
            "- Do not replace baseline ESS/sec with targeted ESS/sec unless a dashboard view is explicitly labeled diagnostic.",
            "",
            "## Smoke contamination",
            f"- Remaining smoke-contaminated targeted cases: {len(smoke)}.",
            "",
            "## Efficiency winners",
            f"- {winner_line}",
        ]
    )
    if not winner_diff.empty:
        lines.extend(bullets(winner_diff.rename(columns={"recommended_efficiency_winner_baseline": "baseline", "recommended_efficiency_winner_targeted": "targeted"}), ["baseline", "targeted"]))
    (out_dir / "reconciliation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reconciliation, smoke = build_reconciliation(args)
    reconciliation.to_csv(args.out_dir / "baseline_targeted_reconciliation.csv", index=False)
    reconciliation[reconciliation["production_rerun_needed"].astype(bool)].to_csv(args.out_dir / "production_rerun_needed.csv", index=False)
    smoke.to_csv(args.out_dir / "smoke_contaminated_cases.csv", index=False)
    write_report(reconciliation, smoke, args.out_dir)
    print(f"wrote reconciliation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
