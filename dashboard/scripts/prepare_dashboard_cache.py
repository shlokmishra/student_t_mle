"""Prepare cached dashboard artifacts for research/dashboard runs.

The script only reads existing outputs and writes compact cache files. It does
not run reference audits, samplers, or KDE machinery.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kde_ref.reference_adapter import DEFAULT_AUDIT_DIR, DEFAULT_BACKENDS, build_posterior_density_grid, mle_sample_cache_path
from models.model_registry import model_validity_rows


CACHE_DIR = Path("results/dashboard_cache/final_production_v1")
REFERENCE_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv")
REFERENCE_DENSITY_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv")
REFERENCE_SUMMARY_CSV = Path("reporting/diagnostic_outputs/model_reference_audit/full_summary/reference_summary.csv")
RUNSET_DIR = Path("results/final_production_v1")
CORRECTNESS_DIR = Path("results/final_production_v1_correctness_audit")
EFFICIENCY_DIR = Path("results/final_production_v1_efficiency_audit_cost_first")
GEOMETRY_DIR = Path("results/final_production_v1_geometry_audit")
RELEASE_INFO_DIR = Path("results/final_production_v1_release_information_audit")
KDE_CORRECTNESS_DIR = Path("results/kde_correctness_audit")
MEETING_PACK_DIR = Path("results/meeting_pack")
RECONCILED_VERDICTS_CSV = MEETING_PACK_DIR / "reconciled_sampler_verdict_table.csv"
FIGURE_DIRS = [
    CORRECTNESS_DIR / "figures",
    EFFICIENCY_DIR / "figures",
    GEOMETRY_DIR / "figures",
    RELEASE_INFO_DIR / "figures",
    KDE_CORRECTNESS_DIR / "figures",
]

EXPECTED_INPUTS = {
    "reference_csv": REFERENCE_CSV,
    "reference_density_csv": REFERENCE_DENSITY_CSV,
    "reference_summary_csv": REFERENCE_SUMMARY_CSV,
    "final_runset_dir": RUNSET_DIR,
    "cost_ledger_csv": RUNSET_DIR / "case_*" / "cost_ledger.csv",
    "posterior_summaries_csv": RUNSET_DIR / "case_*" / "posterior_summaries.csv",
    "chain_samples_csv": RUNSET_DIR / "case_*" / "chain_samples.csv",
    "posterior_accuracy_csv": CORRECTNESS_DIR / "posterior_agreement.csv",
    "cost_efficiency_csv": EFFICIENCY_DIR / "efficiency_summary.csv",
    "method_rankings_csv": EFFICIENCY_DIR / "method_winners.csv",
    "rattle_diagnostics_csv": CORRECTNESS_DIR / "rattle_geometry_diagnostics.csv",
    "suspicious_cases_csv": CORRECTNESS_DIR / "suspicious_sampler_cases.csv",
    "sampler_final_verdicts_csv": CORRECTNESS_DIR / "final_sampler_verdict_table.csv",
    "sampler_diagnostic_coverage_csv": CORRECTNESS_DIR / "diagnostic_coverage_table.csv",
    "sampler_decision_memo_md": CORRECTNESS_DIR / "sampler_correctness_decision_memo.md",
    "release_information_report_md": RELEASE_INFO_DIR / "release_information_report.md",
}

DEFAULT_VIEWS = [
    {
        "view_id": "student_k2",
        "model": "student_t",
        "k": 2.0,
        "n_values": [10, 20, 50],
        "methods": ["raw weighted-MC", "KDE scott", "KDE SJ_transform", "Gibbs", "RATTLE"],
        "warning": "",
    },
    {
        "view_id": "student_k3",
        "model": "student_t",
        "k": 3.0,
        "n_values": [10, 20, 50],
        "methods": ["raw weighted-MC", "KDE scott", "KDE SJ_transform", "Gibbs", "RATTLE"],
        "warning": "",
    },
    {
        "view_id": "logistic",
        "model": "logistic",
        "k": np.nan,
        "n_values": [10, 20, 50],
        "methods": ["raw weighted-MC", "KDE scott", "KDE SJ_transform", "Gibbs", "RATTLE"],
        "warning": "",
    },
    {
        "view_id": "laplace",
        "model": "laplace",
        "k": np.nan,
        "n_values": [11, 21, 51],
        "methods": ["raw weighted-MC", "KDE scott", "KDE SJ_transform", "Gibbs"],
        "warning": "Laplace RATTLE is not applicable; odd-n default uses the unique deterministic sample median target.",
    },
    {
        "view_id": "student_k1",
        "model": "student_t",
        "k": 1.0,
        "n_values": [10, 20, 50],
        "methods": ["raw weighted-MC", "KDE scott", "KDE SJ_transform", "Gibbs", "RATTLE"],
        "warning": "Student k=1,n=10 unresolved.",
    },
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_runset_csv(filename: str, runset_dir: Path = RUNSET_DIR) -> pd.DataFrame:
    root_path = runset_dir / filename
    if root_path.exists():
        return read_csv(root_path)
    frames = []
    for path in sorted(runset_dir.glob(f"case_*/{filename}")):
        frame = read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def input_exists(path: Path) -> bool:
    text = str(path)
    if any(char in text for char in "*?["):
        return bool(list(Path(".").glob(text)))
    return path.exists()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def values(df: pd.DataFrame, column: str) -> list:
    if df.empty or column not in df.columns:
        return []
    vals = df[column].dropna().unique().tolist()
    return sorted(vals, key=lambda value: str(value))


def infer_data_level(reference: pd.DataFrame, cost_ledger: pd.DataFrame) -> str:
    if reference.empty or cost_ledger.empty:
        return "partial"
    b_full = "B" in reference.columns and reference["B"].dropna().astype(float).max() >= 100000
    iter_full = "iterations" in cost_ledger.columns and cost_ledger["iterations"].dropna().astype(float).max() >= 10000
    ref_models = reference["model"].astype(str) if "model" in reference.columns else pd.Series(dtype=str)
    smooth_n = set(reference[ref_models.isin(["student_t", "logistic"])]["n"].dropna().astype(int).unique())
    laplace_n = set(reference[ref_models.eq("laplace")]["n"].dropna().astype(int).unique())
    n_full = smooth_n >= {10, 20, 50} and laplace_n >= {11, 21, 51}
    if b_full and iter_full and n_full:
        return "full"
    if iter_full:
        return "medium"
    if not reference.empty and not cost_ledger.empty:
        return "smoke"
    return "partial"


def filter_view(df: pd.DataFrame, view: dict) -> pd.DataFrame:
    if df.empty or "model" not in df.columns or "n" not in df.columns:
        return pd.DataFrame()
    out = df[df["model"].astype(str).eq(view["model"])].copy()
    out = out[out["n"].astype(int).isin(view["n_values"])]
    if view["model"] == "student_t" and "k" in out.columns:
        out = out[np.isclose(out["k"].astype(float), float(view["k"]))]
    return out


def figure_index() -> pd.DataFrame:
    rows = []
    for figure_dir in FIGURE_DIRS:
        if not figure_dir.exists():
            continue
        for path in sorted(figure_dir.glob("*.png")):
            rows.append(
                {
                    "figure": path.name,
                    "path": str(path),
                    "exists": path.exists(),
                    "audit": figure_dir.parent.name,
                    "kind": "posterior_overlay" if path.name.startswith("posterior_overlay") else "trace" if path.name.startswith("trace") else "metric",
                }
            )
    return pd.DataFrame(rows)


def build_views(reference: pd.DataFrame, accuracy: pd.DataFrame, cost: pd.DataFrame, rankings: pd.DataFrame, figures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    figure_names = figures["figure"].astype(str).tolist() if not figures.empty and "figure" in figures.columns else []
    for view in DEFAULT_VIEWS:
        ref_view = filter_view(reference, view)
        accuracy_view = filter_view(accuracy, view)
        cost_view = filter_view(cost, view)
        ranking_view = filter_view(rankings, view)
        prefix = "student_t_k" + str(int(view["k"])) if view["model"] == "student_t" else view["model"] + "_kna"
        relevant_figures = [name for name in figure_names if prefix in name]
        if view["model"] == "student_t":
            relevant_figures = [name for name in figure_names if f"student_t_k{int(view['k'])}" in name]
        rows.append(
            {
                "view_id": view["view_id"],
                "model": view["model"],
                "k": view["k"],
                "n_values": ",".join(map(str, view["n_values"])),
                "methods": ", ".join(view["methods"]),
                "reference_rows": int(len(ref_view)),
                "posterior_accuracy_rows": int(len(accuracy_view)),
                "cost_efficiency_rows": int(len(cost_view)),
                "method_ranking_rows": int(len(ranking_view)),
                "figure_count": int(len(relevant_figures)),
                "figure_paths": ";".join(relevant_figures),
                "warning": view["warning"],
                "available": bool(len(ref_view) > 0 and (view["model"] == "laplace" or len(accuracy_view) > 0)),
            }
        )
    return pd.DataFrame(rows)


def thinned_chain(chain: pd.DataFrame, thin: int = 20) -> pd.DataFrame:
    keep_cols = ["model", "k", "n", "method", "seed", "iteration", "mu", "is_burn_in"]
    if chain.empty:
        return pd.DataFrame(columns=keep_cols)
    out = chain.copy()
    if "iteration" in out.columns:
        out = out[out["iteration"].astype(int) % int(thin) == 0]
    return out[[col for col in keep_cols if col in out.columns]]


def cached_student_k2_density_grid(grid_size: int = 2000, seeds: tuple[int, ...] = (123, 456, 789)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    status_rows = []
    for n in [10, 20, 50]:
        for seed in seeds:
            path = mle_sample_cache_path(k=2.0, n=int(n), B=100000, seed=int(seed), audit_dir=DEFAULT_AUDIT_DIR)
            status = {
                "model": "student_t",
                "k": 2.0,
                "n": int(n),
                "B": 100000,
                "seed": int(seed),
                "sample_cache_path": str(path),
                "sample_cache_exists": path.exists(),
                "density_cache_status": "missing samples",
            }
            if path.exists():
                z_samples = np.asarray(np.load(path)["z_samples"], dtype=float)
                full_backends = tuple(backend for backend in DEFAULT_BACKENDS if backend != "t_abram")
                density_parts = []
                if full_backends:
                    density_parts.append(
                        build_posterior_density_grid(
                            z_samples=z_samples,
                            k=2.0,
                            n=int(n),
                            mu_star=0.0,
                            prior_mean=0.0,
                            prior_std=10.0,
                            B=100000,
                            seed=int(seed),
                            backends=full_backends,
                            grid_size=int(grid_size),
                            bound_multiplier=5.0,
                        )
                    )
                if "t_abram" in DEFAULT_BACKENDS:
                    rng = np.random.default_rng(20260603 + 1000 * int(seed) + int(n))
                    cap = min(z_samples.size, 5000)
                    idx = np.sort(rng.choice(z_samples.size, size=cap, replace=False))
                    t_samples = z_samples[idx]
                    t_density = build_posterior_density_grid(
                        z_samples=t_samples,
                        k=2.0,
                        n=int(n),
                        mu_star=0.0,
                        prior_mean=0.0,
                        prior_std=10.0,
                        B=int(z_samples.size),
                        seed=int(seed),
                        backends=("t_abram",),
                        grid_size=int(grid_size),
                        bound_multiplier=5.0,
                    )
                    t_density["density_sample_size"] = int(cap)
                    t_density["density_note"] = "t_abram cached from deterministic subsample; scott/SJ_transform use full B=100000 samples."
                    density_parts.append(t_density)
                density = pd.concat(density_parts, ignore_index=True) if density_parts else pd.DataFrame()
                if "density_sample_size" not in density.columns:
                    density["density_sample_size"] = int(z_samples.size)
                density["density_sample_size"] = density["density_sample_size"].fillna(int(z_samples.size))
                if "density_note" not in density.columns:
                    density["density_note"] = ""
                density["density_note"] = density["density_note"].fillna("")
                density["model"] = "student_t"
                density["source_file"] = str(path)
                rows.append(density)
                status["density_cache_status"] = "ready"
                status["density_note"] = "scott/SJ_transform full B=100000; t_abram deterministic subsample."
            status_rows.append(status)
    density_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return density_df, pd.DataFrame(status_rows)


def sampler_density_cache(chain: pd.DataFrame, reference: pd.DataFrame, grid_size: int = 2000, thin: int = 1) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    rows = []
    source = RUNSET_DIR / "case_*" / "chain_samples.csv"
    post = chain.copy()
    if "is_burn_in" in post.columns:
        post = post[~post["is_burn_in"].astype(bool)]
    if "iteration" in post.columns and thin > 1:
        post = post[post["iteration"].astype(int) % int(thin) == 0]
    if "mu_star" not in post.columns:
        post["mu_star"] = 0.0
    if "k" not in post.columns:
        post["k"] = np.nan
    raw = reference[reference["estimator_type"].astype(str).isin(["raw_weighted_mc", "raw_mc_interval_reference"])].copy()
    for keys, part in post.groupby(["model", "method", "n", "k", "mu_star", "seed"], dropna=False, sort=False):
        model, method, n, k, mu_star, seed = keys
        samples = part["mu"].dropna().to_numpy(dtype=float)
        if samples.size < 2:
            continue
        ref = raw[(raw["model"].astype(str).eq(str(model))) & (raw["n"].astype(int).eq(int(n)))]
        if str(model) == "student_t" and "k" in ref.columns and np.isfinite(float(k)):
            ref = ref[np.isclose(ref["k"].astype(float), float(k))]
        if not ref.empty:
            ref_row = ref.iloc[0]
            width = max(float(ref_row["q975"] - ref_row["q025"]), float(ref_row["sd"]), 1e-3)
            lo = float(ref_row["q025"] - 0.25 * width)
            hi = float(ref_row["q975"] + 0.25 * width)
        else:
            q025, q975 = np.quantile(samples, [0.025, 0.975])
            width = max(float(q975 - q025), float(np.std(samples)), 1e-3)
            lo = float(q025 - 0.25 * width)
            hi = float(q975 + 0.25 * width)
        grid = np.linspace(lo, hi, int(grid_size))
        try:
            kde = stats.gaussian_kde(samples)
            density = np.asarray(kde(grid), dtype=float)
            density_method = "gaussian_kde_scott"
        except Exception:
            bins = min(max(int(np.sqrt(samples.size)), 30), 120)
            hist, edges = np.histogram(samples, bins=bins, range=(lo, hi), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            density = np.interp(grid, centers, hist, left=0.0, right=0.0)
            density_method = "histogram_fallback"
        integral = float(np.trapezoid(density, grid))
        if integral > 0:
            density = density / integral
        cdf = np.concatenate([[0.0], np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2.0)])
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        rows.append(
            pd.DataFrame(
                {
                    "model": str(model),
                    "method": str(method),
                    "estimator_type": "sampler_density",
                    "backend": str(method),
                    "n": int(n),
                    "k": float(k) if np.isfinite(float(k)) else np.nan,
                    "mu_star": float(mu_star),
                    "seed": int(seed),
                    "B": np.nan,
                    "mu": grid,
                    "density": density,
                    "cdf": cdf,
                    "posterior_integral_check": float(np.trapezoid(density, grid)),
                    "plot_grid_lo": lo,
                    "plot_grid_hi": hi,
                    "plot_grid_size": int(grid_size),
                    "density_method": density_method,
                    "density_note": "Display-only KDE smoothing of Gibbs/RATTLE chain samples.",
                    "source_file": str(source),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    reference = read_csv(REFERENCE_CSV)
    reference_density = read_csv(REFERENCE_DENSITY_CSV)
    reference_summary = read_csv(REFERENCE_SUMMARY_CSV)
    cost_ledger = read_runset_csv("cost_ledger.csv")
    diagnostics = read_runset_csv("transition_diagnostics.csv")
    posterior_summaries = read_runset_csv("posterior_summaries.csv")
    chain = read_runset_csv("chain_samples.csv")
    accuracy = read_csv(CORRECTNESS_DIR / "posterior_agreement.csv")
    cost_efficiency = read_csv(EFFICIENCY_DIR / "efficiency_summary.csv")
    rankings = read_csv(EFFICIENCY_DIR / "method_winners.csv")
    rattle_diagnostics = read_csv(CORRECTNESS_DIR / "rattle_geometry_diagnostics.csv")
    suspicious = read_csv(CORRECTNESS_DIR / "suspicious_sampler_cases.csv")
    student_diag = read_csv(CORRECTNESS_DIR / "target_mismatch_diagnostics.csv")
    standalone_sampler_verdicts = read_csv(CORRECTNESS_DIR / "final_sampler_verdict_table.csv")
    reconciled_sampler_verdicts = read_csv(RECONCILED_VERDICTS_CSV)
    sampler_final_verdicts = reconciled_sampler_verdicts if not reconciled_sampler_verdicts.empty else standalone_sampler_verdicts
    sampler_coverage = read_csv(CORRECTNESS_DIR / "diagnostic_coverage_table.csv")
    sampler_suspicious = read_csv(CORRECTNESS_DIR / "suspicious_sampler_cases.csv")
    validity = pd.DataFrame(model_validity_rows())
    figures = figure_index()
    views = build_views(reference, accuracy, cost_efficiency, rankings, figures)
    stale_density_reason = ""
    if not reference.empty and not reference_density.empty and "B" in reference.columns and "B" in reference_density.columns:
        reference_b = pd.to_numeric(reference["B"], errors="coerce").max()
        density_b = pd.to_numeric(reference_density["B"], errors="coerce").max()
        if pd.notna(reference_b) and pd.notna(density_b) and density_b < reference_b:
            stale_density_reason = (
                f"Ignored reference density grid because it is preview B={int(density_b)} "
                f"while reference summaries are B={int(reference_b)}."
            )
            reference_density = pd.DataFrame()

    if reference_density.empty:
        posterior_density, density_status = cached_student_k2_density_grid()
    else:
        posterior_density = reference_density.copy()
        if "B_used" not in posterior_density.columns:
            posterior_density["B_used"] = pd.to_numeric(posterior_density.get("B", np.nan), errors="coerce")
        if "density_sample_size" not in posterior_density.columns:
            posterior_density["density_sample_size"] = posterior_density["B_used"]
        posterior_density["density_sample_capped"] = pd.to_numeric(
            posterior_density["density_sample_size"], errors="coerce"
        ) < pd.to_numeric(posterior_density.get("B", np.nan), errors="coerce")
        if "density_note" not in posterior_density.columns:
            posterior_density["density_note"] = ""
        posterior_density.loc[
            posterior_density["backend"].astype(str).eq("t_abram") & posterior_density["density_sample_capped"].fillna(False),
            "density_note",
        ] = "t_abram is adaptive and expensive; cached t_abram curve is capped for visualization only."
        density_status = (
            posterior_density.groupby(["model", "k", "n", "B", "seed"], dropna=False, as_index=False)
            .agg(
                sample_cache_path=("source_file", "first"),
                sample_cache_exists=("source_file", lambda value: True),
                density_cache_status=("backend", lambda value: "ready"),
                B_used=("B_used", "max"),
                t_abram_capped=("density_sample_capped", "max"),
                density_note=("density_note", lambda value: " ".join(sorted({str(item) for item in value if str(item).strip()})) or "loaded from reference_all_models_density_grid.csv"),
            )
        )
    sampler_density = sampler_density_cache(chain, reference, thin=20)

    cache_files = {
        "reference_cache.csv": reference,
        "reference_summary_cache.csv": reference_summary,
        "posterior_density_cache.csv": posterior_density,
        "density_cache_status.csv": density_status,
        "sampler_density_cache.csv": sampler_density,
        "posterior_comparison_cache.csv": accuracy,
        "cost_ledger_cache.csv": cost_ledger,
        "diagnostic_summary_cache.csv": diagnostics,
        "posterior_summaries_cache.csv": posterior_summaries,
        "cost_efficiency_cache.csv": cost_efficiency,
        "method_rankings_cache.csv": rankings,
        "rattle_diagnostics_cache.csv": rattle_diagnostics,
        "suspicious_cases_cache.csv": suspicious,
        "model_validity_cache.csv": validity,
        "student_k1_n10_diagnostic_cache.csv": student_diag,
        "sampler_final_verdict_cache.csv": sampler_final_verdicts,
        "sampler_standalone_final_verdict_cache.csv": standalone_sampler_verdicts,
        "sampler_diagnostic_coverage_cache.csv": sampler_coverage,
        "sampler_suspicious_cases_cache.csv": sampler_suspicious,
        "figure_index.csv": figures,
        "dashboard_views_cache.csv": views,
        "chain_samples_thinned_cache.csv": thinned_chain(chain),
    }
    for filename, df in cache_files.items():
        write_csv(df, CACHE_DIR / filename)
    if (CORRECTNESS_DIR / "sampler_correctness_report.md").exists():
        (CACHE_DIR / "executive_summary_cache.md").write_text(
            (CORRECTNESS_DIR / "sampler_correctness_report.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    if (CORRECTNESS_DIR / "target_mismatch_diagnostics.csv").exists():
        (CACHE_DIR / "student_k1_n10_diagnostic_cache.md").write_text(
            "Student k=1,n=10 is unresolved in the final production correctness audit. "
            "Use this case only as an unresolved caveat.\n",
            encoding="utf-8",
        )
    if (CORRECTNESS_DIR / "sampler_correctness_decision_memo.md").exists():
        (CACHE_DIR / "sampler_correctness_decision_memo_cache.md").write_text(
            (CORRECTNESS_DIR / "sampler_correctness_decision_memo.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    files_found = [name for name, path in EXPECTED_INPUTS.items() if input_exists(path)]
    files_missing = [str(path) for name, path in EXPECTED_INPUTS.items() if not input_exists(path)]
    row_counts = {filename: int(len(df)) for filename, df in cache_files.items()}
    warnings = []
    if files_missing:
        warnings.append("Some expected source artifacts are missing.")
    if stale_density_reason:
        warnings.append(stale_density_reason)
    if not student_diag.empty:
        warnings.append("Student k=1,n=10 unresolved; see student_k1_n10_diagnostic.md.")
    else:
        warnings.append("Student k=1,n=10 diagnostic cache is missing or empty.")
    if not validity.empty:
        laplace_rattle = validity[(validity["model"].eq("laplace")) & (validity["method"].astype(str).str.contains("rattle", case=False, na=False))]
        if not laplace_rattle.empty:
            warnings.append("Laplace RATTLE is marked not applicable.")

    data_level = infer_data_level(reference, cost_ledger)
    dashboard_ready = (
        data_level == "full"
        and row_counts["reference_cache.csv"] > 0
        and row_counts["posterior_comparison_cache.csv"] > 0
        and row_counts["cost_ledger_cache.csv"] > 0
        and row_counts["model_validity_cache.csv"] > 0
        and row_counts["student_k1_n10_diagnostic_cache.csv"] > 0
        and bool(views["available"].all())
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "source_runset": "final_production_v1",
        "data_level": data_level,
        "reference_csv_path": str(REFERENCE_CSV),
        "runset_dir": str(RUNSET_DIR),
        "correctness_dir": str(CORRECTNESS_DIR),
        "efficiency_dir": str(EFFICIENCY_DIR),
        "geometry_dir": str(GEOMETRY_DIR),
        "release_information_dir": str(RELEASE_INFO_DIR),
        "sampler_verdict_source": str(RECONCILED_VERDICTS_CSV if not reconciled_sampler_verdicts.empty else CORRECTNESS_DIR / "final_sampler_verdict_table.csv"),
        "files_found": files_found,
        "files_missing": files_missing,
        "row_counts": row_counts,
        "models_available": values(reference if not reference.empty else accuracy, "model"),
        "k_values_available": values(reference, "k"),
        "n_values_available": values(reference if not reference.empty else cost_ledger, "n"),
        "methods_available": values(accuracy if not accuracy.empty else cost_ledger, "method"),
        "seeds_available": values(cost_ledger if not cost_ledger.empty else reference, "seed"),
        "laplace_rattle_status": "not_applicable",
        "student_k1_n10_status": "unresolved" if not student_diag.empty else "missing",
        "warnings": warnings,
        "dashboard_ready": bool(dashboard_ready),
    }
    (CACHE_DIR / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    status_lines = [
        "# Dashboard Cache",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- data_level: {data_level}",
        f"- dashboard_ready: {dashboard_ready}",
        f"- cache_dir: `{CACHE_DIR}`",
        "",
        "## Warnings",
        "",
        *[f"- {warning}" for warning in warnings],
        "",
        "## Missing Inputs",
        "",
        *([f"- `{path}`" for path in files_missing] if files_missing else ["- none"]),
        "",
        "## Row Counts",
        "",
        *[f"- {name}: {count}" for name, count in sorted(row_counts.items())],
    ]
    (CACHE_DIR / "cache_status.md").write_text("\n".join(status_lines), encoding="utf-8")
    print(f"dashboard_ready={dashboard_ready}")
    print(f"data_level={data_level}")
    print(f"wrote {CACHE_DIR}")


if __name__ == "__main__":
    main()
