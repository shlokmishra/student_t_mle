"""Check that the dashboard cache is complete and scientifically labeled."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CACHE_DIR = Path("results/dashboard_cache")
CHECK_PATH = CACHE_DIR / "cache_check.json"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def check(results: list[dict], name: str, ok: bool, detail: str = "", level: str | None = None) -> None:
    status = level or ("OK" if ok else "MISSING")
    results.append({"check": name, "status": status, "detail": detail})
    print(f"{status:<8} {name} {detail}")


def has_view(views: pd.DataFrame, view_id: str) -> bool:
    if views.empty or "view_id" not in views.columns or "available" not in views.columns:
        return False
    rows = views[views["view_id"].astype(str).eq(view_id)]
    if rows.empty:
        return False
    return bool(rows["available"].astype(bool).iloc[0])


def page_has_no_long_auto_run(page: Path) -> bool:
    if not page.exists():
        return False
    text = page.read_text(encoding="utf-8")
    risky = ["subprocess.run", "subprocess.Popen", "os.system", "os.popen"]
    return not any(item in text for item in risky)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    manifest_path = CACHE_DIR / "cache_manifest.json"
    check(results, "cache_manifest.json exists", manifest_path.exists(), str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    check(results, "dashboard_ready is true", bool(manifest.get("dashboard_ready")), str(manifest.get("dashboard_ready")))

    reference = read_csv(CACHE_DIR / "reference_cache.csv")
    posterior_density = read_csv(CACHE_DIR / "posterior_density_cache.csv")
    sampler_density = read_csv(CACHE_DIR / "sampler_density_cache.csv")
    posterior = read_csv(CACHE_DIR / "posterior_comparison_cache.csv")
    cost = read_csv(CACHE_DIR / "cost_ledger_cache.csv")
    cost_eff = read_csv(CACHE_DIR / "cost_efficiency_cache.csv")
    validity = read_csv(CACHE_DIR / "model_validity_cache.csv")
    student_diag = read_csv(CACHE_DIR / "student_k1_n10_diagnostic_cache.csv")
    views = read_csv(CACHE_DIR / "dashboard_views_cache.csv")

    check(results, "reference cache exists and nonempty", not reference.empty, f"rows={len(reference)}")
    check(results, "posterior KDE density cache exists and nonempty", not posterior_density.empty, f"rows={len(posterior_density)}")
    check(results, "sampler density cache exists and nonempty", not sampler_density.empty, f"rows={len(sampler_density)}")
    check(results, "posterior comparison cache exists and nonempty", not posterior.empty, f"rows={len(posterior)}")
    check(results, "cost cache exists and nonempty", not cost.empty, f"rows={len(cost)}")
    check(results, "analysis report cache exists and nonempty", not cost_eff.empty, f"rows={len(cost_eff)}")
    check(results, "model validity cache exists", not validity.empty, f"rows={len(validity)}")
    for view_id in ["student_k2", "student_k3", "logistic", "laplace"]:
        check(results, f"{view_id} view available", has_view(views, view_id))
    check(results, "Student k=1,n=10 diagnostic exists", not student_diag.empty, f"rows={len(student_diag)}")

    laplace_rattle_ok = False
    laplace_gibbs_ok = False
    laplace_warning_ok = False
    if not validity.empty:
        laplace_rattle = validity[
            validity["model"].astype(str).eq("laplace")
            & validity["method"].astype(str).str.contains("rattle", case=False, na=False)
        ]
        laplace_rattle_ok = not laplace_rattle.empty and (
            laplace_rattle.get("rattle_applicable", pd.Series([""])).astype(str).str.lower().isin(["false", "nan", ""]).all()
            or laplace_rattle.get("implementation_exists", pd.Series([True])).astype(str).str.lower().isin(["false"]).any()
        )
        laplace_gibbs = validity[
            validity["model"].astype(str).eq("laplace")
            & validity["method"].astype(str).str.contains("gibbs", case=False, na=False)
        ]
        laplace_gibbs_ok = not laplace_gibbs.empty and laplace_gibbs["target_description"].astype(str).str.contains(
            "median_interval_contains_mu_star", case=False, na=False
        ).any()
        laplace_warning_ok = validity["warnings"].fillna("").astype(str).str.contains(
            "np.median|not directly comparable|median interval", case=False, regex=True
        ).any() or validity.get("warning", pd.Series([""])).fillna("").astype(str).str.contains(
            "np.median|not directly comparable|median interval", case=False, regex=True
        ).any()
    check(results, "Laplace RATTLE marked not_applicable", laplace_rattle_ok)
    check(results, "Laplace Gibbs target is median_interval_contains_mu_star", laplace_gibbs_ok)
    check(results, "Laplace deterministic np.median warning exists", laplace_warning_ok)

    pages = [
        Path("app.py"),
        Path("pages/1_Posterior_Comparison.py"),
        Path("pages/2_Cost_Audit.py"),
        Path("pages/3_Model_Validity_Audit.py"),
        Path("pages/4_Analysis_Report.py"),
    ]
    for page in pages:
        check(results, f"{page} does not auto-run long audits", page_has_no_long_auto_run(page), str(page))

    CHECK_PATH.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    missing = [row for row in results if row["status"] == "MISSING"]
    warnings = [row for row in results if row["status"] == "WARNING"]
    print(f"wrote {CHECK_PATH}")
    if missing:
        raise SystemExit(1)
    if warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
