"""Check Streamlit dashboard files and generated comparison data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HEALTH_PATH = ROOT / "results" / "analysis_pipeline" / "dashboard_health.json"
PAGES = [
    ROOT / "dashboard" / "app.py",
    ROOT / "dashboard" / "pages" / "1_Posterior_Comparison.py",
    ROOT / "dashboard" / "pages" / "2_Cost_Audit.py",
    ROOT / "dashboard" / "pages" / "3_Model_Validity_Audit.py",
    ROOT / "dashboard" / "pages" / "4_Analysis_Report.py",
    ROOT / "dashboard" / "pages" / "5_KDE_Correctness.py",
    ROOT / "dashboard" / "pages" / "6_Sampler_Correctness.py",
    ROOT / "dashboard" / "pages" / "7_Efficiency.py",
    ROOT / "dashboard" / "pages" / "8_Geometry.py",
]
REFERENCE_FILES = {
    "full": ROOT / "reporting" / "diagnostic_outputs" / "model_reference_audit" / "reference_all_models.csv",
    "smoke": ROOT / "reporting" / "diagnostic_outputs" / "model_reference_audit" / "reference_all_models_smoke.csv",
}
COST_DIRS = {
    "full": ROOT / "results" / "cost_audit",
    "medium": ROOT / "results" / "cost_audit_medium",
    "smoke": ROOT / "results" / "cost_audit_smoke",
    "multiseed": ROOT / "results" / "cost_audit_multiseed",
}
COMMON_COST_FILES = ["cost_ledger.csv", "posterior_summaries.csv", "diagnostic_summary.csv", "chain_samples.csv"]
ANALYSIS_REPORT_FILES = [
    ROOT / "results" / "analysis_report" / "executive_summary.md",
    ROOT / "results" / "analysis_report" / "posterior_accuracy.csv",
    ROOT / "results" / "analysis_report" / "cost_efficiency.csv",
    ROOT / "results" / "analysis_report" / "method_rankings.csv",
    ROOT / "results" / "analysis_report" / "suspicious_cases.csv",
]
EFFICIENCY_AUDIT_FILES = [
    ROOT / "results" / "efficiency_audit" / "efficiency_report.md",
    ROOT / "results" / "efficiency_audit" / "efficiency_summary.csv",
    ROOT / "results" / "efficiency_audit" / "functional_ess.csv",
    ROOT / "results" / "efficiency_audit" / "cost_decomposition.csv",
    ROOT / "results" / "efficiency_audit" / "method_winners.csv",
    ROOT / "results" / "efficiency_audit" / "rattle_movement_diagnostics.csv",
    ROOT / "results" / "efficiency_audit" / "caveat_efficiency_cases.csv",
    ROOT / "results" / "efficiency_audit" / "timing_warnings.csv",
]
GEOMETRY_AUDIT_FILES = [
    ROOT / "results" / "geometry_audit" / "geometry_report.md",
    ROOT / "results" / "geometry_audit" / "geometry_summary.csv",
    ROOT / "results" / "geometry_audit" / "latent_tail_geometry.csv",
    ROOT / "results" / "geometry_audit" / "geometry_conditioned_posterior.csv",
    ROOT / "results" / "geometry_audit" / "rattle_geometry_explanation.csv",
    ROOT / "results" / "geometry_audit" / "gibbs_geometry_explanation.csv",
    ROOT / "results" / "geometry_audit" / "branch_exploration.csv",
    ROOT / "results" / "geometry_audit" / "rattle_tail_failure_analysis.csv",
    ROOT / "results" / "geometry_audit" / "gibbs_local_move_analysis.csv",
    ROOT / "results" / "geometry_audit" / "geometry_win_loss_table.csv",
    ROOT / "results" / "geometry_audit" / "missing_geometry_diagnostics.csv",
    ROOT / "results" / "geometry_audit" / "unresolved_geometry_cases.csv",
]


def line(status: str, label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"{status:<8} {label}{suffix}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def check_pages() -> dict:
    rows = {}
    for path in PAGES:
        exists = path.exists()
        rows[str(path.relative_to(ROOT))] = {"exists": exists}
        line("OK" if exists else "MISSING", str(path.relative_to(ROOT)))
    return rows


def check_registry() -> dict:
    try:
        from models.model_registry import MODEL_REGISTRY, model_validity_rows

        models = sorted(MODEL_REGISTRY.keys())
        validity = pd.DataFrame(model_validity_rows())
        line("OK", "model registry", ",".join(models))
        return {
            "exists": True,
            "models": models,
            "validity_rows": int(len(validity)),
            "laplace_rattle_not_applicable": bool(
                not validity[
                    validity["model"].eq("laplace")
                    & validity["method"].eq("rattle")
                    & validity["rattle_applicable"].eq(False)
                ].empty
            ),
        }
    except Exception as exc:
        line("MISSING", "model registry", str(exc))
        return {"exists": False, "error": str(exc)}


def check_reference_files() -> dict:
    out = {}
    for level, path in REFERENCE_FILES.items():
        df = read_csv(path)
        info = {"path": str(path), "exists": path.exists(), "rows": int(len(df))}
        if not df.empty:
            info["models"] = sorted(df["model"].dropna().astype(str).unique().tolist()) if "model" in df.columns else []
            info["targets"] = sorted(df["target_description"].dropna().astype(str).unique().tolist()) if "target_description" in df.columns else []
            info["laplace_has_interval_reference"] = bool(
                "target_description" in df.columns
                and df["target_description"].astype(str).eq("median_interval_contains_mu_star").any()
            )
            info["laplace_has_np_median_reference"] = bool(
                "target_description" in df.columns
                and df["target_description"].astype(str).isin(
                    ["deterministic_median_equals_mu_star", "deterministic_np_median_equals_mu_star"]
                ).any()
            )
        out[level] = info
        line("OK" if path.exists() else "MISSING", f"{level} reference CSV", f"rows={info['rows']}")
    return out


def check_cost_dirs() -> dict:
    out = {}
    for level, directory in COST_DIRS.items():
        files = {name: directory / name for name in COMMON_COST_FILES}
        missing = [name for name, path in files.items() if not path.exists()]
        ledger = read_csv(files["cost_ledger.csv"])
        chains = read_csv(files["chain_samples.csv"])
        info = {
            "path": str(directory),
            "exists": directory.exists(),
            "missing_files": missing,
            "ledger_rows": int(len(ledger)),
            "chain_rows": int(len(chains)),
        }
        if not ledger.empty:
            info["models"] = sorted(ledger["model"].dropna().astype(str).unique().tolist()) if "model" in ledger.columns else []
            info["methods"] = sorted(ledger["method"].dropna().astype(str).unique().tolist()) if "method" in ledger.columns else []
            info["run_status"] = sorted(ledger["run_status"].dropna().astype(str).unique().tolist()) if "run_status" in ledger.columns else []
            info["n_values"] = sorted(ledger["n"].dropna().astype(int).unique().tolist()) if "n" in ledger.columns else []
            info["laplace_rattle_not_applicable_row"] = bool(
                "model" in ledger.columns
                and "method" in ledger.columns
                and "rattle_status" in ledger.columns
                and ledger["model"].astype(str).eq("laplace").any()
                and ledger[
                    ledger["model"].astype(str).eq("laplace")
                    & ledger["method"].astype(str).eq("rattle")
                    & ledger["rattle_status"].astype(str).eq("not_applicable")
                ].shape[0]
                > 0
            )
            info["target_descriptions"] = sorted(ledger["target_description"].dropna().astype(str).unique().tolist()) if "target_description" in ledger.columns else []
        out[level] = info
        status = "OK" if not missing else "MISSING"
        line(status, f"{level} cost audit", f"ledger_rows={info['ledger_rows']} missing={','.join(missing) if missing else 'none'}")
    return out


def current_data_level(reference: dict, cost: dict) -> dict:
    ref_level = next((level for level in ["full", "smoke"] if reference.get(level, {}).get("exists")), "missing")
    cost_level = next((level for level in ["full", "medium", "smoke"] if cost.get(level, {}).get("ledger_rows", 0) > 0), "missing")
    return {"reference": ref_level, "cost": cost_level}


def check_analysis_report() -> dict:
    out = {}
    for path in ANALYSIS_REPORT_FILES:
        exists = path.exists()
        out[str(path.relative_to(ROOT))] = {"exists": exists}
        line("OK" if exists else "MISSING", str(path.relative_to(ROOT)))
    return out


def check_efficiency_audit() -> dict:
    out = {}
    for path in EFFICIENCY_AUDIT_FILES:
        exists = path.exists()
        rows = int(len(read_csv(path))) if path.suffix == ".csv" and exists else 0
        out[str(path.relative_to(ROOT))] = {"exists": exists, "rows": rows}
        detail = f"rows={rows}" if path.suffix == ".csv" and exists else ""
        line("OK" if exists else "MISSING", str(path.relative_to(ROOT)), detail)
    return out


def check_geometry_audit() -> dict:
    out = {}
    for path in GEOMETRY_AUDIT_FILES:
        exists = path.exists()
        rows = int(len(read_csv(path))) if path.suffix == ".csv" and exists else 0
        out[str(path.relative_to(ROOT))] = {"exists": exists, "rows": rows}
        detail = f"rows={rows}" if path.suffix == ".csv" and exists else ""
        line("OK" if exists else "MISSING", str(path.relative_to(ROOT)), detail)
    missing_path = ROOT / "results" / "geometry_audit" / "missing_geometry_diagnostics.csv"
    if missing_path.exists():
        missing = read_csv(missing_path)
        high = missing[missing.get("severity", pd.Series(dtype=str)).astype(str).isin(["high", "medium"])]
        if not high.empty:
            line("WARN", "geometry diagnostics incomplete", f"medium/high missing={len(high)}")
            out["missing_medium_high"] = int(len(high))
    targeted_dir = ROOT / "results" / "targeted_validation_runs"
    if not targeted_dir.exists():
        line("WARN", "targeted validation runset", "results/targeted_validation_runs is not present yet")
        out["targeted_validation_present"] = False
    else:
        out["targeted_validation_present"] = True
    return out


def main() -> None:
    print("Dashboard Data Check")
    print("====================")
    try:
        import streamlit as st

        line("OK", "Streamlit", st.__version__)
        streamlit = {"exists": True, "version": st.__version__}
    except Exception as exc:
        line("MISSING", "Streamlit", str(exc))
        streamlit = {"exists": False, "error": str(exc)}

    health = {
        "streamlit": streamlit,
        "pages": check_pages(),
        "registry": check_registry(),
        "reference": check_reference_files(),
        "cost": check_cost_dirs(),
        "analysis_report": check_analysis_report(),
        "efficiency_audit": check_efficiency_audit(),
        "geometry_audit": check_geometry_audit(),
    }
    health["current_data_level"] = current_data_level(health["reference"], health["cost"])
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_PATH.write_text(json.dumps(health, indent=2, sort_keys=True), encoding="utf-8")
    line("OK", "dashboard_health.json", str(HEALTH_PATH.relative_to(ROOT)))


if __name__ == "__main__":
    main()
