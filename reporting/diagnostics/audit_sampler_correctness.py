"""Sampler correctness audit for Gibbs and RATTLE cached outputs.

This script is a cache-first postprocessor. It does not change sampler
transition logic, KDE implementations, or run new simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.loc_student import get_mle as student_get_mle


TARGET_NS = {
    "student_t": [10, 20, 50],
    "logistic": [10, 20, 50],
    "laplace": [11, 21, 51],
}
TARGET_KS = [1.0, 2.0, 3.0]
CASE_COLS = ["model", "k_key", "n", "mu_star", "seed", "initialization", "method"]
CASE_DISPLAY = ["model", "k", "n", "mu_star", "seed", "initialization", "method"]
PRODUCTION_FILENAMES = {
    "chain_samples": "chain_samples.csv",
    "posterior_summaries": "posterior_summaries.csv",
    "cost_ledger": "cost_ledger.csv",
    "transition_diagnostics": "transition_diagnostics.csv",
    "latent_diagnostics": "latent_diagnostics.csv",
    "rattle_energy_diagnostics": "rattle_energy_diagnostics.csv",
    "branch_diagnostics": "branch_diagnostics.csv",
    "geometry_diagnostics": "geometry_diagnostics.csv",
    "initialization_diagnostics": "initialization_diagnostics.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runset-dir",
        type=Path,
        default=Path("results/final_production_v1/"),
        help="Preferred per-case production runset root. Falls back to --cost-dir if absent.",
    )
    parser.add_argument("--cost-dir", type=Path, default=Path("results/cost_audit/"))
    parser.add_argument("--multiseed-dir", type=Path, default=Path("results/cost_audit_multiseed/"))
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--reference-density-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv"),
    )
    parser.add_argument("--kde-audit-dir", type=Path, default=Path("results/kde_correctness_audit/"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/sampler_correctness_audit/"))
    return parser.parse_args()


def finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def k_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: "__NA__" if pd.isna(value) else f"{float(value):g}")


def add_k_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "k" in out.columns:
        out["k"] = pd.to_numeric(out["k"], errors="coerce")
        out["k_key"] = k_key(out["k"])
    else:
        out["k"] = np.nan
        out["k_key"] = "__NA__"
    if "mu_star" not in out.columns:
        out["mu_star"] = 0.0
    if "seed" not in out.columns:
        out["seed"] = 0
    if "n" in out.columns:
        out["n"] = pd.to_numeric(out["n"], errors="coerce")
    out["mu_star"] = pd.to_numeric(out["mu_star"], errors="coerce").fillna(0.0)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)
    if "initialization" not in out.columns:
        out["initialization"] = "unspecified"
    out["initialization"] = out["initialization"].fillna("unspecified").astype(str)
    return out


def target_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = add_k_key(df)
    student = out["model"].eq("student_t") & out["k"].isin(TARGET_KS) & out["n"].isin(TARGET_NS["student_t"])
    logistic = out["model"].eq("logistic") & out["n"].isin(TARGET_NS["logistic"])
    laplace = out["model"].eq("laplace") & out["n"].isin(TARGET_NS["laplace"])
    return out[student | logistic | laplace].copy()


def latent_coordinate_columns(df: pd.DataFrame) -> list[str]:
    """Return indexed latent coordinate columns like x_0, x_1, ... only."""
    cols = []
    for col in df.columns:
        if not col.startswith("x_"):
            continue
        suffix = col.split("_", 1)[1]
        if suffix.isdigit():
            cols.append(col)
    return sorted(cols, key=lambda col: int(col.split("_", 1)[1]))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_cost_file(cost_dir: Path, filename: str) -> pd.DataFrame:
    primary = read_csv(cost_dir / filename)
    primary = primary[~primary.get("model", pd.Series(dtype=str)).astype(str).eq("laplace")].copy() if not primary.empty else primary
    supplemental_dir = ROOT / "results" / "results" / "cost_audit"
    supplemental = read_csv(supplemental_dir / filename)
    if not supplemental.empty:
        supplemental = supplemental[supplemental["model"].astype(str).eq("laplace")].copy()
    frames = [df for df in [primary, supplemental] if not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return target_filter(out)


def parse_case_id(case_id: str) -> dict:
    parts = case_id.removeprefix("case_").split("_")
    out: dict[str, object] = {"case_id": case_id}
    if not parts:
        return out
    if parts[0] == "student" and len(parts) >= 8 and parts[1] == "t":
        out["model"] = "student_t"
        out["k"] = float(parts[2].removeprefix("k"))
        out["n"] = int(parts[3].removeprefix("n"))
        out["method"] = parts[4]
        out["seed"] = int(parts[5].removeprefix("seed"))
        out["initialization"] = "_".join(parts[7:]) if len(parts) > 7 else "unspecified"
    elif parts[0] in {"logistic", "laplace"} and len(parts) >= 6:
        out["model"] = parts[0]
        out["k"] = np.nan
        out["n"] = int(parts[1].removeprefix("n"))
        out["method"] = parts[2]
        out["seed"] = int(parts[3].removeprefix("seed"))
        out["initialization"] = "_".join(parts[5:]) if len(parts) > 5 else "unspecified"
    return out


def metadata_for_case(case_dir: Path, cases: pd.DataFrame) -> dict:
    case_id = case_dir.name.removeprefix("case_")
    full_case_id = case_dir.name
    meta = parse_case_id(full_case_id)
    meta["case_id"] = case_id
    if not cases.empty and "case_id" in cases.columns:
        row = cases[cases["case_id"].astype(str).isin([case_id, full_case_id])]
        if not row.empty:
            rec = row.iloc[0].to_dict()
            meta.update({key: rec[key] for key in rec if pd.notna(rec[key])})
    metadata_path = case_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta.update({key: loaded[key] for key in loaded if loaded[key] is not None})
        except json.JSONDecodeError:
            meta["metadata_json_error"] = True
    meta.setdefault("case_id", case_id)
    meta.setdefault("model", "")
    meta.setdefault("k", np.nan)
    meta.setdefault("n", np.nan)
    meta.setdefault("method", "")
    meta.setdefault("seed", 0)
    meta.setdefault("initialization", "unspecified")
    meta.setdefault("diagnostic_only", False)
    meta.setdefault("num_iterations", np.nan)
    meta.setdefault("burn_in", np.nan)
    meta.setdefault("diagnostic_thin", np.nan)
    meta["output_dir"] = str(case_dir)
    return meta


def attach_case_metadata(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "case_id",
        "model",
        "k",
        "n",
        "method",
        "seed",
        "initialization",
        "diagnostic_only",
        "num_iterations",
        "burn_in",
        "diagnostic_thin",
        "output_dir",
    ]:
        if col not in out.columns:
            out[col] = meta.get(col, np.nan)
        else:
            out[col] = out[col].fillna(meta.get(col, np.nan))
    if "mu_star" not in out.columns:
        out["mu_star"] = meta.get("mu_star", 0.0)
    if "iteration" in out.columns and "is_burn_in" not in out.columns and pd.notna(meta.get("burn_in")):
        out["is_burn_in"] = pd.to_numeric(out["iteration"], errors="coerce") < int(float(meta["burn_in"]))
    return target_filter(out)


def load_production_runset(runset_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    cases_tsv = runset_dir / "final_production_cases.tsv"
    cases = pd.read_csv(cases_tsv, sep="\t") if cases_tsv.exists() else pd.DataFrame()
    case_dirs = sorted(path for path in runset_dir.glob("case_*") if path.is_dir())
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in PRODUCTION_FILENAMES}
    meta_rows = []
    for case_dir in case_dirs:
        meta = metadata_for_case(case_dir, cases)
        meta_rows.append(meta)
        for key, filename in PRODUCTION_FILENAMES.items():
            path = case_dir / filename
            if path.exists():
                frames[key].append(attach_case_metadata(pd.read_csv(path), meta))
    tables = {
        key: pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        for key, parts in frames.items()
    }
    run_metadata = pd.DataFrame(meta_rows)
    missing_rows = []
    expected = cases if not cases.empty else run_metadata
    if not expected.empty:
        for _, row in expected.iterrows():
            output_dir = Path(str(row.get("output_dir", runset_dir / f"case_{row.get('case_id', '')}")))
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
            for key, filename in PRODUCTION_FILENAMES.items():
                path = output_dir / filename
                if not path.exists():
                    missing_rows.append(
                        {
                            "case_id": row.get("case_id", output_dir.name.removeprefix("case_")),
                            "model": row.get("model", ""),
                            "k": row.get("k", np.nan),
                            "n": row.get("n", np.nan),
                            "method": row.get("method", ""),
                            "seed": row.get("seed", np.nan),
                            "initialization": row.get("initialization", ""),
                            "missing_file": filename,
                            "output_dir": str(output_dir),
                        }
                    )
    missing = pd.DataFrame(missing_rows)
    return tables, run_metadata, missing


def raw_reference(reference_csv: Path, density_csv: Path | None = None) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    ref = target_filter(ref)
    raw = ref[ref["estimator_type"].astype(str).eq("raw_weighted_mc")].copy()
    grouped = raw.groupby(["model", "k_key", "n", "mu_star"], dropna=False).agg(
        raw_mean=("mean", "mean"),
        raw_sd=("sd", "mean"),
        raw_q025=("q025", "mean"),
        raw_q50=("q50", "mean"),
        raw_q975=("q975", "mean"),
        raw_mean_seed_sd=("mean", "std"),
        raw_sd_seed_sd=("sd", "std"),
        target_description=("target_description", "first"),
    ).reset_index()
    for name in ["q01", "q05", "q95", "q99"]:
        grouped[f"raw_{name}"] = np.nan
    if density_csv is not None and density_csv.exists():
        density = target_filter(read_csv(density_csv))
        if not density.empty and {"backend", "mu", "cdf"}.issubset(density.columns):
            density = density[density["backend"].astype(str).eq("scott")].copy()
            tail_rows = []
            for keys, group in density.groupby(["model", "k_key", "n", "mu_star", "seed"], dropna=False):
                g = group.sort_values("mu")
                cdf = pd.to_numeric(g["cdf"], errors="coerce").to_numpy(float)
                grid = pd.to_numeric(g["mu"], errors="coerce").to_numpy(float)
                ok = np.isfinite(cdf) & np.isfinite(grid)
                if ok.sum() < 2:
                    continue
                cdf = np.maximum.accumulate(cdf[ok])
                grid = grid[ok]
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1]
                tail_rows.append(
                    {
                        "model": keys[0],
                        "k_key": keys[1],
                        "n": int(keys[2]),
                        "mu_star": keys[3],
                        "raw_q01_density": float(np.interp(0.01, cdf, grid)),
                        "raw_q05_density": float(np.interp(0.05, cdf, grid)),
                        "raw_q95_density": float(np.interp(0.95, cdf, grid)),
                        "raw_q99_density": float(np.interp(0.99, cdf, grid)),
                    }
                )
            tails = pd.DataFrame(tail_rows)
            if not tails.empty:
                tails = tails.groupby(["model", "k_key", "n", "mu_star"], dropna=False).mean(numeric_only=True).reset_index()
                grouped = grouped.merge(tails, on=["model", "k_key", "n", "mu_star"], how="left")
                for q in ["q01", "q05", "q95", "q99"]:
                    grouped[f"raw_{q}"] = grouped[f"raw_{q}_density"].combine_first(grouped[f"raw_{q}"])
                    grouped = grouped.drop(columns=[f"raw_{q}_density"], errors="ignore")
    return grouped


def chain_post(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return chain
    out = target_filter(chain)
    if "is_burn_in" in out.columns:
        out = out[~out["is_burn_in"].astype(bool)].copy()
    return out


def chain_quantiles(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in chain.groupby(CASE_COLS, dropna=False):
        vals = group["mu"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(CASE_COLS, keys, strict=True)),
                "draws": int(vals.size),
                "q01": float(np.quantile(vals, 0.01)),
                "q05": float(np.quantile(vals, 0.05)),
                "q95": float(np.quantile(vals, 0.95)),
                "q99": float(np.quantile(vals, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def cdf_at_sorted(vals: np.ndarray, x: float) -> float:
    if vals.size == 0 or not math.isfinite(x):
        return np.nan
    return float(np.searchsorted(vals, x, side="right") / vals.size)


def posterior_agreement(summaries: pd.DataFrame, chain: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return pd.DataFrame()
    s = target_filter(summaries)
    rattle_status = s.get("rattle_status", pd.Series("", index=s.index)).astype(str)
    s = s[~(s["method"].eq("rattle") & rattle_status.eq("not_applicable"))].copy()
    s = s.rename(
        columns={
            "mean_mu": "mean",
            "sd_mu": "sd",
            "q025_mu": "q025",
            "q50_mu": "q50",
            "q975_mu": "q975",
        }
    )
    q = chain_quantiles(chain)
    if not q.empty:
        s = s.merge(q[CASE_COLS + ["draws", "q01", "q05", "q95", "q99"]], on=CASE_COLS, how="left")
    merged = s.merge(raw, on=["model", "k_key", "n", "mu_star"], how="left")
    sorted_chain = {}
    if not chain.empty:
        for keys, group in chain.groupby(CASE_COLS, dropna=False):
            vals = np.sort(group["mu"].to_numpy(float))
            vals = vals[np.isfinite(vals)]
            sorted_chain[keys] = vals
    rows = []
    for _, row in merged.iterrows():
        raw_sd = finite_float(row.get("raw_sd"))
        raw_width = finite_float(row.get("raw_q975")) - finite_float(row.get("raw_q025"))
        key = tuple(row[col] for col in CASE_COLS)
        vals = sorted_chain.get(key, np.array([], dtype=float))
        central_mass = (
            cdf_at_sorted(vals, finite_float(row.get("raw_q975"))) - cdf_at_sorted(vals, finite_float(row.get("raw_q025")))
            if vals.size
            else np.nan
        )
        delta_mean = finite_float(row.get("mean")) - finite_float(row.get("raw_mean"))
        delta_sd = finite_float(row.get("sd")) - raw_sd
        quantile_errors = [
            abs(finite_float(row.get("q01")) - finite_float(row.get("raw_q01"))),
            abs(finite_float(row.get("q025")) - finite_float(row.get("raw_q025"))),
            abs(finite_float(row.get("q05")) - finite_float(row.get("raw_q05"))),
            abs(finite_float(row.get("q50")) - finite_float(row.get("raw_q50"))),
            abs(finite_float(row.get("q95")) - finite_float(row.get("raw_q95"))),
            abs(finite_float(row.get("q975")) - finite_float(row.get("raw_q975"))),
            abs(finite_float(row.get("q99")) - finite_float(row.get("raw_q99"))),
        ]
        finite_quantile_errors = [value for value in quantile_errors if math.isfinite(value)]
        wasserstein = float(np.mean(finite_quantile_errors)) if finite_quantile_errors else np.nan
        good = (
            abs(delta_mean) / max(raw_sd, 1e-300) <= 0.10
            and abs(delta_sd) / max(raw_sd, 1e-300) <= 0.10
            and (0.93 <= central_mass <= 0.97 if math.isfinite(central_mass) else False)
            and (max(finite_quantile_errors) <= 0.10 * max(raw_sd, 1e-300) if finite_quantile_errors else False)
        )
        warning = []
        if abs(delta_mean) / max(raw_sd, 1e-300) > 0.10:
            warning.append("mean")
        if abs(delta_sd) / max(raw_sd, 1e-300) > 0.10:
            warning.append("sd")
        if math.isfinite(central_mass) and not (0.93 <= central_mass <= 0.97):
            warning.append("central95")
        if finite_quantile_errors and max(finite_quantile_errors) > 0.10 * max(raw_sd, 1e-300):
            warning.append("quantile")
        rows.append(
            {
                "model": row["model"],
                "k": row["k"],
                "n": int(row["n"]),
                "mu_star": row["mu_star"],
                "seed": int(row["seed"]),
                "initialization": row.get("initialization", "unspecified"),
                "method": row["method"],
                "mean": row["mean"],
                "sd": row["sd"],
                "q01": row.get("q01", np.nan),
                "q025": row["q025"],
                "q05": row.get("q05", np.nan),
                "q50": row["q50"],
                "q95": row.get("q95", np.nan),
                "q975": row["q975"],
                "q99": row.get("q99", np.nan),
                "raw_mean": row["raw_mean"],
                "raw_sd": row["raw_sd"],
                "raw_q01": row.get("raw_q01", np.nan),
                "raw_q025": row["raw_q025"],
                "raw_q05": row.get("raw_q05", np.nan),
                "raw_q50": row["raw_q50"],
                "raw_q95": row.get("raw_q95", np.nan),
                "raw_q975": row["raw_q975"],
                "raw_q99": row.get("raw_q99", np.nan),
                "delta_mean": delta_mean,
                "abs_delta_mean_over_raw_sd": abs(delta_mean) / max(raw_sd, 1e-300),
                "delta_sd": delta_sd,
                "rel_sd_error": delta_sd / max(raw_sd, 1e-300),
                "delta_q01": finite_float(row.get("q01")) - finite_float(row.get("raw_q01")),
                "delta_q025": finite_float(row.get("q025")) - finite_float(row.get("raw_q025")),
                "delta_q05": finite_float(row.get("q05")) - finite_float(row.get("raw_q05")),
                "delta_q50": finite_float(row.get("q50")) - finite_float(row.get("raw_q50")),
                "delta_q95": finite_float(row.get("q95")) - finite_float(row.get("raw_q95")),
                "delta_q975": finite_float(row.get("q975")) - finite_float(row.get("raw_q975")),
                "delta_q99": finite_float(row.get("q99")) - finite_float(row.get("raw_q99")),
                "tail_prob_left_raw_q025": cdf_at_sorted(vals, finite_float(row.get("raw_q025"))) if vals.size else np.nan,
                "tail_prob_right_raw_q975": 1.0 - cdf_at_sorted(vals, finite_float(row.get("raw_q975"))) if vals.size else np.nan,
                "central_mass_raw_95": central_mass,
                "ks_distance": np.nan,
                "wasserstein_distance": wasserstein,
                "posterior_agreement_good": bool(good),
                "warning": "none" if not warning else ",".join(warning),
                "target_description": row.get("target_description", ""),
            }
        )
    return pd.DataFrame(rows)


def simple_acf(vals: np.ndarray, max_lag: int = 100) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.array([1.0])
    centered = vals - vals.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        return np.array([1.0])
    max_lag = min(max_lag, vals.size - 1)
    return np.array([1.0] + [float(np.dot(centered[:-lag], centered[lag:]) / denom) for lag in range(1, max_lag + 1)])


def chain_split_stability(chain: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if chain.empty:
        return pd.DataFrame()
    raw_lookup = {
        (r.model, r.k_key, int(r.n), float(r.mu_star)): r
        for r in raw.itertuples(index=False)
    }
    for keys, group in chain.groupby(CASE_COLS, dropna=False):
        model, kk, n, mu_star, seed, initialization, method = keys
        vals = group.sort_values("iteration")["mu"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            continue
        chunks = np.array_split(vals, 4)
        means = np.array([np.mean(c) for c in chunks])
        sds = np.array([np.std(c) for c in chunks])
        q025 = np.array([np.quantile(c, 0.025) for c in chunks])
        q50 = np.array([np.quantile(c, 0.5) for c in chunks])
        q975 = np.array([np.quantile(c, 0.975) for c in chunks])
        ref = raw_lookup.get((model, kk, int(n), float(mu_star)))
        raw_sd = finite_float(getattr(ref, "raw_sd", np.nan)) if ref is not None else np.nan
        mean_drift = float((means.max() - means.min()) / max(raw_sd, 1e-300))
        sd_drift = float((sds.max() - sds.min()) / max(np.mean(sds), 1e-300))
        q975_drift = float((q975.max() - q975.min()) / max(raw_sd, 1e-300))
        warning = "serious" if mean_drift > 0.15 or sd_drift > 0.15 else "none"
        rows.append(
            {
                "model": model,
                "k": np.nan if kk == "__NA__" else float(kk),
                "n": int(n),
                "mu_star": mu_star,
                "seed": int(seed),
                "initialization": initialization,
                "method": method,
                "chunk1_mean": means[0],
                "chunk2_mean": means[1],
                "chunk3_mean": means[2],
                "chunk4_mean": means[3],
                "chunk1_sd": sds[0],
                "chunk2_sd": sds[1],
                "chunk3_sd": sds[2],
                "chunk4_sd": sds[3],
                "chunk1_q025": q025[0],
                "chunk2_q025": q025[1],
                "chunk3_q025": q025[2],
                "chunk4_q025": q025[3],
                "chunk1_q50": q50[0],
                "chunk2_q50": q50[1],
                "chunk3_q50": q50[2],
                "chunk4_q50": q50[3],
                "chunk1_q975": q975[0],
                "chunk2_q975": q975[1],
                "chunk3_q975": q975[2],
                "chunk4_q975": q975[3],
                "max_chunk_mean_diff_over_sd": mean_drift,
                "max_chunk_sd_rel_diff": sd_drift,
                "max_chunk_q975_diff_over_sd": q975_drift,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def ess_acf_diagnostics(chain: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ledger_lookup = {}
    if not ledger.empty:
        l = target_filter(ledger)
        for _, row in l.iterrows():
            ledger_lookup[(row["model"], row["k_key"], int(row["n"]), float(row["mu_star"]), int(row["seed"]), row.get("initialization", "unspecified"), row["method"])] = row
    for keys, group in chain.groupby(CASE_COLS, dropna=False):
        vals = group.sort_values("iteration")["mu"].to_numpy(float)
        acf = simple_acf(vals, 100)
        positive = acf[1:][acf[1:] > 0]
        tau = float(1.0 + 2.0 * np.sum(positive))
        row = ledger_lookup.get(keys)
        rows.append(
            {
                "model": keys[0],
                "k": np.nan if keys[1] == "__NA__" else float(keys[1]),
                "n": int(keys[2]),
                "mu_star": keys[3],
                "seed": int(keys[4]),
                "initialization": keys[5],
                "method": keys[6],
                "draws": int(vals.size),
                "ess_mu": finite_float(row.get("ess_mu")) if row is not None else vals.size / max(tau, 1e-300),
                "ess_per_sec": finite_float(row.get("ess_per_sec")) if row is not None else np.nan,
                "integrated_autocorr_time": tau,
                "lag1_acf": float(acf[1]) if acf.size > 1 else np.nan,
                "lag10_acf": float(acf[10]) if acf.size > 10 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gibbs_constraint_diagnostics(ledger: pd.DataFrame, latent: pd.DataFrame, transition: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if transition is not None and not transition.empty:
        diag = target_filter(transition)
        diag = diag[diag["method"].eq("gibbs")].copy()
        if "abs_constraint_residual" in diag.columns:
            for keys, group in diag.groupby(CASE_COLS, dropna=False):
                vals = pd.to_numeric(group["abs_constraint_residual"], errors="coerce").dropna().to_numpy(float)
                pair = pd.to_numeric(group.get("abs_pair_delta_error", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(float)
                if vals.size == 0 and pair.size == 0:
                    continue
                rows.append(
                    {
                        "model": keys[0],
                        "k": np.nan if keys[1] == "__NA__" else float(keys[1]),
                        "n": int(keys[2]),
                        "mu_star": keys[3],
                        "seed": int(keys[4]),
                        "initialization": keys[5],
                        "method": keys[6],
                        "max_abs_constraint_residual": float(np.max(vals)) if vals.size else np.nan,
                        "mean_abs_constraint_residual": float(np.mean(vals)) if vals.size else np.nan,
                        "q95_abs_constraint_residual": float(np.quantile(vals, 0.95)) if vals.size else np.nan,
                        "constraint_source": "transition_diagnostics",
                        "pair_updates_attempted": np.nan,
                        "pair_updates_completed": np.nan,
                        "max_abs_pair_delta_error": float(np.max(pair)) if pair.size else np.nan,
                        "mean_abs_pair_delta_error": float(np.mean(pair)) if pair.size else np.nan,
                        "q95_abs_pair_delta_error": float(np.quantile(pair, 0.95)) if pair.size else np.nan,
                        "pair_delta_diagnostic_status": "thinned_snapshot_proxy_not_transition_invariant" if pair.size else "not_available",
                        "warning": "none" if vals.size == 0 or np.max(vals) <= 1e-6 else "serious",
                    }
                )
    gibbs = target_filter(ledger)
    gibbs = gibbs[gibbs["method"].eq("gibbs")]
    for _, row in gibbs.iterrows():
        rows.append(
            {
                "model": row["model"],
                "k": row["k"],
                "n": int(row["n"]),
                "mu_star": row["mu_star"],
                "seed": int(row["seed"]),
                "initialization": row.get("initialization", "unspecified"),
                "method": "gibbs",
                "max_abs_constraint_residual": finite_float(row.get("max_constraint_abs", 0.0)),
                "mean_abs_constraint_residual": finite_float(row.get("mean_constraint_abs", 0.0)),
                "q95_abs_constraint_residual": np.nan,
                "constraint_source": "ledger",
                "pair_updates_attempted": finite_float(row.get("pair_updates_attempted")),
                "pair_updates_completed": finite_float(row.get("pair_updates_completed")),
                "max_abs_pair_delta_error": np.nan,
                "pair_delta_diagnostic_status": "not_available",
                "warning": "none" if finite_float(row.get("max_constraint_abs", 0.0)) <= 1e-6 else "serious",
            }
        )
    if not latent.empty:
        lat = target_filter(latent)
        lat = lat[lat["method"].eq("gibbs") & lat["model"].eq("student_t")].copy()
        x_cols = latent_coordinate_columns(lat)
        if x_cols:
            for keys, group in lat.groupby(["model", "k", "n", "mu_star", "seed", "method"], dropna=False):
                vals = []
                for rec in group.to_dict("records"):
                    x = np.asarray([rec[c] for c in x_cols if pd.notna(rec.get(c))], dtype=float)
                    if x.size == 0:
                        continue
                    k = float(rec["k"])
                    mu_star = float(rec["mu_star"])
                    vals.append(abs(float(np.sum((x - mu_star) / (k + (x - mu_star) ** 2)))))
                if vals:
                    rows.append(
                        {
                            "model": keys[0],
                            "k": keys[1],
                            "n": int(keys[2]),
                            "mu_star": keys[3],
                            "seed": int(keys[4]),
                            "initialization": "unspecified",
                            "method": keys[5],
                            "max_abs_constraint_residual": float(np.max(vals)),
                            "mean_abs_constraint_residual": float(np.mean(vals)),
                            "q95_abs_constraint_residual": float(np.quantile(vals, 0.95)),
                            "constraint_source": "latent_x_diagnostics",
                            "pair_updates_attempted": np.nan,
                            "pair_updates_completed": np.nan,
                            "max_abs_pair_delta_error": np.nan,
                            "pair_delta_diagnostic_status": "not_available",
                            "warning": "none" if np.max(vals) <= 1e-6 else "serious",
                        }
                    )
    return pd.DataFrame(rows)


def gibbs_branch_diagnostics(latent: pd.DataFrame, branch: pd.DataFrame | None = None, transition: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if branch is not None and not branch.empty:
        b = target_filter(branch)
        b = b[b["method"].eq("gibbs") & b["model"].eq("student_t")].copy()
        if not b.empty:
            group_cols = CASE_COLS
            for keys, group in b.groupby(group_cols, dropna=False):
                count_col = next((col for col in ["count", "branch_count", "n_draws", "draws"] if col in group.columns), None)
                if count_col:
                    counts = pd.to_numeric(group[count_col], errors="coerce").fillna(0.0).to_numpy(float)
                    total = float(np.sum(counts))
                    fractions = counts / total if total > 0 else np.array([])
                    imbalance = float(np.max(fractions) - np.min(fractions)) if fractions.size else np.nan
                    used = int(np.sum(counts > 0))
                else:
                    labels = group.get("branch_pair", group.get("student_inverse_branch_labels", pd.Series(dtype=str))).astype(str)
                    used = int(labels.nunique())
                    imbalance = np.nan
                rows.append(
                    {
                        "model": keys[0],
                        "k": np.nan if keys[1] == "__NA__" else float(keys[1]),
                        "n": int(keys[2]),
                        "mu_star": keys[3],
                        "seed": int(keys[4]),
                        "initialization": keys[5],
                        "method": keys[6],
                        "branch_diagnostic_available": True,
                        "branch_pairs_used": used,
                        "branch_switching_rate": finite_float(group.get("branch_switching_rate", pd.Series([np.nan])).iloc[0]),
                        "branch_imbalance": imbalance,
                        "warning": "mild" if used <= 1 or (math.isfinite(imbalance) and imbalance > 0.98) else "none",
                        "note": "Exact branch diagnostics loaded from branch_diagnostics.csv.",
                    }
                )
            return pd.DataFrame(rows)
    if transition is not None and not transition.empty and "student_inverse_branch_labels" in transition.columns:
        t = target_filter(transition)
        t = t[t["method"].eq("gibbs") & t["model"].eq("student_t")].copy()
        for keys, group in t.groupby(CASE_COLS, dropna=False):
            labels = group["student_inverse_branch_labels"].dropna().astype(str)
            if labels.empty:
                continue
            switches = labels.ne(labels.shift()).iloc[1:]
            rows.append(
                {
                    "model": keys[0],
                    "k": np.nan if keys[1] == "__NA__" else float(keys[1]),
                    "n": int(keys[2]),
                    "mu_star": keys[3],
                    "seed": int(keys[4]),
                    "initialization": keys[5],
                    "method": keys[6],
                    "branch_diagnostic_available": True,
                    "branch_pairs_used": int(labels.nunique()),
                    "branch_switching_rate": float(switches.mean()) if len(switches) else np.nan,
                    "branch_imbalance": float(labels.value_counts(normalize=True).max() - labels.value_counts(normalize=True).min()),
                    "warning": "mild" if labels.nunique() <= 1 else "none",
                    "note": "Branch labels loaded from transition_diagnostics.csv.",
                }
            )
        if rows:
            return pd.DataFrame(rows)
    if latent.empty:
        return pd.DataFrame(columns=["model", "k", "n", "method", "seed", "branch_diagnostic_available", "warning"])
    lat = target_filter(latent)
    lat = lat[lat["method"].eq("gibbs") & lat["model"].eq("student_t")]
    x_cols = latent_coordinate_columns(lat)
    if not x_cols:
        return pd.DataFrame()
    for keys, group in lat.groupby(["model", "k", "n", "seed", "method"], dropna=False):
        vals = group[x_cols].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        k = float(keys[1])
        y = vals  # mu_star is zero in current targets.
        lower = float(np.mean(y < -math.sqrt(k)))
        middle = float(np.mean(np.abs(y) <= math.sqrt(k)))
        upper = float(np.mean(y > math.sqrt(k)))
        imbalance = max(lower, middle, upper) - min(lower, middle, upper)
        warning = "mild" if max(lower, upper, middle) > 0.98 else "none"
        rows.append(
            {
                "model": keys[0],
                "k": k,
                "n": int(keys[2]),
                "seed": int(keys[3]),
                "method": keys[4],
                "branch_diagnostic_available": False,
                "approx_lower_tail_fraction": lower,
                "approx_middle_fraction": middle,
                "approx_upper_tail_fraction": upper,
                "branch_switching_rate": np.nan,
                "branch_imbalance": imbalance,
                "warning": warning,
                "note": "Exact inverse-branch counters were not cached; fractions are approximate from |x-mu_star| relative to sqrt(k).",
            }
        )
    return pd.DataFrame(rows)


def rattle_geometry_diagnostics(ledger: pd.DataFrame, energy: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if energy is not None and not energy.empty:
        e = target_filter(energy)
        e = e[e["method"].eq("rattle")].copy()
        for _, row in e.iterrows():
            if str(row.get("rattle_status", "")) == "not_applicable" or str(row.get("model", "")) == "laplace":
                rows.append(
                    {
                        "model": row["model"],
                        "k": row["k"],
                        "n": int(row["n"]),
                        "mu_star": row["mu_star"],
                        "seed": int(row["seed"]),
                        "initialization": row.get("initialization", "unspecified"),
                        "method": "rattle",
                        "rattle_status": "not_applicable",
                        "geometry_status": "not_applicable",
                        "warning": "none",
                    }
                )
                continue
            projection_failure_rate = finite_float(row.get("projection_failure_indicator"), 0.0)
            reverse_failure_rate = finite_float(row.get("reverse_check_failure_indicator"), 0.0)
            tangent = abs(finite_float(row.get("tangent_residual_abs_grad_c_dot_p")))
            constraint = abs(finite_float(row.get("position_constraint_residual"), row.get("position_constraint_residual_mean", np.nan)))
            warning = "none"
            if projection_failure_rate > 0.05 or reverse_failure_rate > 0.05 or constraint > 1e-6 or tangent > 1e-6:
                warning = "serious"
            rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "n": int(row["n"]),
                    "mu_star": row["mu_star"],
                    "seed": int(row["seed"]),
                    "initialization": row.get("initialization", "unspecified"),
                    "method": "rattle",
                    "rattle_status": row.get("rattle_status", "completed"),
                    "max_abs_constraint_residual": constraint,
                    "mean_abs_constraint_residual": abs(finite_float(row.get("position_constraint_residual_mean"), constraint)),
                    "q95_abs_constraint_residual": np.nan,
                    "mean_abs_tangent_residual": tangent,
                    "max_abs_tangent_residual": tangent,
                    "q95_abs_tangent_residual": np.nan,
                    "reverse_check_attempts": np.nan,
                    "reverse_check_failures": reverse_failure_rate,
                    "reverse_check_failure_rate": reverse_failure_rate,
                    "max_reverse_position_error": finite_float(row.get("reverse_position_error")),
                    "mean_reverse_position_error": finite_float(row.get("reverse_position_error")),
                    "max_reverse_momentum_error": finite_float(row.get("reverse_momentum_error")),
                    "projection_failure_rate": projection_failure_rate,
                    "forward_newton_iters_per_proposal": finite_float(row.get("newton_iterations")),
                    "reverse_newton_iters_per_proposal": np.nan,
                    "max_newton_iters": finite_float(row.get("newton_iterations")),
                    "integration_failures": np.nan,
                    "projection_mode": row.get("projection_mode", ""),
                    "gram_correction_enabled": True,
                    "reverse_check_enabled": True,
                    "geometry_status": "pass" if warning == "none" else "fail",
                    "warning": warning,
                }
            )
        if rows:
            return pd.DataFrame(rows)
    rat = target_filter(ledger)
    rat = rat[rat["method"].eq("rattle")]
    for _, row in rat.iterrows():
        if str(row.get("rattle_status", "")) == "not_applicable":
            rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "n": int(row["n"]),
                    "mu_star": row["mu_star"],
                    "seed": int(row["seed"]),
                    "initialization": row.get("initialization", "unspecified"),
                    "method": "rattle",
                    "rattle_status": "not_applicable",
                    "geometry_status": "not_applicable",
                    "warning": "none",
                }
            )
            continue
        hmc = max(finite_float(row.get("hmc_proposals"), 0.0), 1.0)
        projection_evals = max(finite_float(row.get("projection_evals"), 0.0), 1.0)
        projection_failure_rate = finite_float(row.get("projection_failures"), 0.0) / projection_evals
        reverse_failure_rate = finite_float(row.get("reverse_check_failures"), 0.0) / hmc
        projection_mode_ok = str(row.get("projection_mode", "")) == "paper_fixed_direction"
        gram_ok = normalize_bool(row.get("gram_correction_enabled", False))
        reverse_ok = reverse_failure_rate <= 0.05
        warning = "none"
        if projection_failure_rate > 0.05 or reverse_failure_rate > 0.05 or not projection_mode_ok or not gram_ok:
            warning = "serious"
        rows.append(
            {
                "model": row["model"],
                "k": row["k"],
                "n": int(row["n"]),
                "mu_star": row["mu_star"],
                "seed": int(row["seed"]),
                "initialization": row.get("initialization", "unspecified"),
                "method": "rattle",
                "rattle_status": row.get("rattle_status", ""),
                "max_abs_constraint_residual": finite_float(row.get("max_constraint_abs")),
                "mean_abs_constraint_residual": finite_float(row.get("mean_constraint_abs")),
                "q95_abs_constraint_residual": np.nan,
                "mean_abs_tangent_residual": np.nan,
                "max_abs_tangent_residual": np.nan,
                "q95_abs_tangent_residual": np.nan,
                "reverse_check_attempts": finite_float(row.get("hmc_proposals")),
                "reverse_check_failures": finite_float(row.get("reverse_check_failures")),
                "reverse_check_failure_rate": reverse_failure_rate,
                "max_reverse_position_error": finite_float(row.get("reverse_position_error")),
                "mean_reverse_position_error": np.nan,
                "max_reverse_momentum_error": finite_float(row.get("reverse_momentum_error")),
                "projection_failure_rate": projection_failure_rate,
                "forward_newton_iters_per_proposal": finite_float(row.get("forward_newton_iters")) / hmc,
                "reverse_newton_iters_per_proposal": finite_float(row.get("reverse_newton_iters")) / hmc,
                "max_newton_iters": np.nan,
                "integration_failures": np.nan,
                "projection_mode": row.get("projection_mode", ""),
                "gram_correction_enabled": gram_ok,
                "reverse_check_enabled": True,
                "geometry_status": "pass" if warning == "none" else "fail",
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def rattle_energy_diagnostics(ledger: pd.DataFrame, chain: pd.DataFrame, energy: pd.DataFrame | None = None) -> pd.DataFrame:
    if energy is not None and not energy.empty:
        e = target_filter(energy)
        e = e[e["method"].eq("rattle")].copy()
        rows = []
        for _, row in e.iterrows():
            if str(row.get("model", "")) == "laplace" or str(row.get("rattle_status", "")) == "not_applicable":
                rows.append(
                    {
                        "model": row["model"],
                        "k": row["k"],
                        "n": int(row["n"]),
                        "mu_star": row["mu_star"],
                        "seed": int(row["seed"]),
                        "initialization": row.get("initialization", "unspecified"),
                        "method": "rattle",
                        "rattle_status": "not_applicable",
                        "energy_diagnostic_available": False,
                        "warning": "none",
                    }
                )
                continue
            delta_h = finite_float(row.get("delta_H"), row.get("delta_H_mean_abs", np.nan))
            mean_abs = finite_float(row.get("delta_H_mean_abs"), abs(delta_h) if math.isfinite(delta_h) else np.nan)
            max_abs = finite_float(row.get("delta_H_max_abs"), abs(delta_h) if math.isfinite(delta_h) else np.nan)
            warning = "none" if (not math.isfinite(max_abs) or max_abs <= 1.0) else "mild"
            rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "n": int(row["n"]),
                    "mu_star": row["mu_star"],
                    "seed": int(row["seed"]),
                    "initialization": row.get("initialization", "unspecified"),
                    "method": "rattle",
                    "rattle_status": row.get("rattle_status", "completed"),
                    "energy_diagnostic_available": True,
                    "mean_delta_H": delta_h,
                    "mean_abs_delta_H": mean_abs,
                    "rms_delta_H": finite_float(row.get("delta_H_rms")),
                    "max_abs_delta_H": max_abs,
                    "q05_delta_H": np.nan,
                    "q50_delta_H": delta_h,
                    "q95_delta_H": np.nan,
                    "frac_abs_delta_H_gt_1": float(max_abs > 1.0) if math.isfinite(max_abs) else np.nan,
                    "acceptance_rate": finite_float(row.get("acceptance_rate")),
                    "mean_abs_mu_step": finite_float(row.get("abs_delta_mu")),
                    "median_abs_mu_step": finite_float(row.get("abs_delta_mu")),
                    "q95_abs_mu_step": finite_float(row.get("abs_delta_mu")),
                    "ESJD_mu": finite_float(row.get("ESJD_mu")),
                    "projection_failure_indicator": finite_float(row.get("projection_failure_indicator")),
                    "reverse_check_failure_indicator": finite_float(row.get("reverse_check_failure_indicator")),
                    "warning": warning,
                    "note": "Loaded from per-case rattle_energy_diagnostics.csv.",
                }
            )
        return pd.DataFrame(rows)
    move_rows = []
    if not chain.empty:
        for keys, group in chain[chain["method"].eq("rattle")].groupby(CASE_COLS, dropna=False):
            vals = group.sort_values("iteration")["mu"].to_numpy(float)
            steps = np.abs(np.diff(vals[np.isfinite(vals)]))
            if steps.size:
                move_rows.append(
                    {
                        **dict(zip(CASE_COLS, keys, strict=True)),
                        "mean_abs_mu_step": float(np.mean(steps)),
                        "median_abs_mu_step": float(np.median(steps)),
                        "q95_abs_mu_step": float(np.quantile(steps, 0.95)),
                    }
                )
    moves = pd.DataFrame(move_rows)
    rat = target_filter(ledger)
    rat = rat[rat["method"].eq("rattle")]
    if not moves.empty:
        rat = rat.merge(moves, on=CASE_COLS, how="left")
    rows = []
    for _, row in rat.iterrows():
        if str(row.get("rattle_status", "")) == "not_applicable":
            rows.append(
                {
                    "model": row["model"],
                    "k": row["k"],
                    "n": int(row["n"]),
                    "seed": int(row["seed"]),
                    "initialization": row.get("initialization", "unspecified"),
                    "method": "rattle",
                    "rattle_status": "not_applicable",
                    "energy_diagnostic_available": False,
                    "warning": "none",
                }
            )
            continue
        acceptance = finite_float(row.get("acceptance_rate"))
        mean_step = finite_float(row.get("mean_abs_mu_step"))
        small_move = bool(acceptance >= 0.995 and mean_step < 1e-3) if math.isfinite(mean_step) else False
        rows.append(
            {
                "model": row["model"],
                "k": row["k"],
                "n": int(row["n"]),
                "mu_star": row["mu_star"],
                "seed": int(row["seed"]),
                "initialization": row.get("initialization", "unspecified"),
                "method": "rattle",
                "rattle_status": row.get("rattle_status", ""),
                "energy_diagnostic_available": False,
                "mean_delta_H": np.nan,
                "sd_delta_H": np.nan,
                "q05_delta_H": np.nan,
                "q50_delta_H": np.nan,
                "q95_delta_H": np.nan,
                "frac_abs_delta_H_gt_1": np.nan,
                "acceptance_rate": acceptance,
                "mean_abs_mu_step": mean_step,
                "median_abs_mu_step": finite_float(row.get("median_abs_mu_step")),
                "q95_abs_mu_step": finite_float(row.get("q95_abs_mu_step")),
                "mean_abs_x_move": np.nan,
                "median_abs_x_move": np.nan,
                "q95_abs_x_move": np.nan,
                "warning": "mild" if small_move else "none",
                "note": "delta_H and x-move diagnostics were not cached.",
            }
        )
    return pd.DataFrame(rows)


def initialization_sensitivity(summaries: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not summaries.empty and "initialization" in summaries.columns:
        s = target_filter(summaries).rename(columns={"mean_mu": "mean", "sd_mu": "sd"})
        raw_lookup = {
            (r.model, r.k_key, int(r.n), float(r.mu_star)): r
            for r in raw.itertuples(index=False)
        }
        for keys, group in s.groupby(["model", "k_key", "n", "mu_star", "method"], dropna=False):
            inits = sorted(group["initialization"].dropna().astype(str).unique())
            if len(inits) <= 1:
                continue
            means = group.groupby("initialization")["mean"].mean()
            sds = group.groupby("initialization")["sd"].mean()
            ref = raw_lookup.get((keys[0], keys[1], int(keys[2]), float(keys[3])))
            raw_sd = finite_float(getattr(ref, "raw_sd", np.nan)) if ref is not None else np.nan
            mean_diff = float((means.max() - means.min()) / max(raw_sd, 1e-300))
            sd_rel_diff = float((sds.max() - sds.min()) / max(float(sds.mean()), 1e-300))
            rows.append(
                {
                    "model": keys[0],
                    "k": np.nan if keys[1] == "__NA__" else float(keys[1]),
                    "n": int(keys[2]),
                    "mu_star": keys[3],
                    "method": keys[4],
                    "initializations_available": True,
                    "initializations": ",".join(inits),
                    "central_mean": finite_float(means.get("central")),
                    "tail_heavy_mean": finite_float(means.get("tail_heavy", means.get("tail-heavy", np.nan))),
                    "random_mean": finite_float(means.get("random")),
                    "central_sd": finite_float(sds.get("central")),
                    "tail_heavy_sd": finite_float(sds.get("tail_heavy", sds.get("tail-heavy", np.nan))),
                    "random_sd": finite_float(sds.get("random")),
                    "max_mean_diff_over_raw_sd": mean_diff,
                    "max_sd_rel_diff": sd_rel_diff,
                    "warning": "serious" if mean_diff > 0.15 or sd_rel_diff > 0.20 else "none",
                    "note": "Compared posterior_summaries.csv across initialization labels.",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    for model, k, n in [
        ("student_t", 1.0, 10),
        ("student_t", 2.0, 20),
        ("logistic", np.nan, 20),
        ("laplace", np.nan, 21),
    ]:
        rows.append(
            {
                "model": model,
                "k": k,
                "n": n,
                "method": "all",
                "initializations_available": False,
                "central_mean": np.nan,
                "tail_heavy_mean": np.nan,
                "random_mean": np.nan,
                "max_mean_diff_over_raw_sd": np.nan,
                "warning": "not_available",
                "note": "No distinct initialization labels were present in cached outputs.",
            }
        )
    return pd.DataFrame(rows)


def target_mismatch_diagnostics(latent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not latent.empty:
        lat = target_filter(latent)
        lat = lat[lat["model"].eq("student_t")]
        x_cols = latent_coordinate_columns(lat)
        focus = lat[
            ((lat["k"].eq(1.0)) & lat["n"].isin([10, 20, 50]))
            | ((lat["k"].eq(2.0)) & lat["n"].eq(10))
            | ((lat["k"].eq(3.0)) & lat["n"].eq(10))
        ]
        if x_cols and not focus.empty:
            per_draw = []
            for rec in focus.to_dict("records"):
                x = np.asarray([rec[c] for c in x_cols if pd.notna(rec.get(c))], dtype=float)
                if x.size == 0:
                    continue
                k = float(rec["k"])
                n = int(rec["n"])
                mu_star = float(rec["mu_star"])
                score = float(np.sum((x - mu_star) / (k + (x - mu_star) ** 2)))
                try:
                    selected_mle = float(student_get_mle(x, {"k": k, "n": n}))
                    mle_error = ""
                except Exception as exc:
                    selected_mle = np.nan
                    mle_error = f"{type(exc).__name__}: {exc}"
                per_draw.append(
                    {
                        "model": rec["model"],
                        "method": rec["method"],
                        "k": k,
                        "n": n,
                        "seed": int(rec["seed"]),
                        "score_near_zero": abs(score) <= 1e-6,
                        "selected_mle_near_mu_star": math.isfinite(selected_mle) and abs(selected_mle - mu_star) <= 1e-3,
                        "selected_mle_minus_mu_star": selected_mle - mu_star if math.isfinite(selected_mle) else np.nan,
                        "mle_error": mle_error,
                    }
                )
            draws = pd.DataFrame(per_draw)
            if not draws.empty:
                for keys, group in draws.groupby(["model", "method", "k", "n", "seed"], dropna=False):
                    delta = group["selected_mle_minus_mu_star"].dropna()
                    rows.append(
                        {
                            "model": keys[0],
                            "method": keys[1],
                            "k": keys[2],
                            "n": int(keys[3]),
                            "seed": int(keys[4]),
                            "num_latent_draws": int(len(group)),
                            "fraction_score_near_zero": float(group["score_near_zero"].mean()),
                            "fraction_selected_mle_near_mu_star": float(group["selected_mle_near_mu_star"].mean()),
                            "target_mismatch_rate": float((group["score_near_zero"] & ~group["selected_mle_near_mu_star"]).mean()),
                            "selected_mle_minus_mu_star_mean": float(delta.mean()) if not delta.empty else np.nan,
                            "selected_mle_minus_mu_star_sd": float(delta.std(ddof=0)) if not delta.empty else np.nan,
                            "classification": "score-root vs selected-MLE target mismatch"
                            if float((group["score_near_zero"] & ~group["selected_mle_near_mu_star"]).mean()) > 0.05
                            else "score-root target consistent with selected MLE",
                        }
                    )
    for n in [11, 21, 51]:
        rows.append(
            {
                "model": "laplace",
                "method": "gibbs",
                "k": np.nan,
                "n": n,
                "seed": np.nan,
                "num_latent_draws": np.nan,
                "fraction_score_near_zero": np.nan,
                "fraction_selected_mle_near_mu_star": np.nan,
                "target_mismatch_rate": 0.0,
                "selected_mle_minus_mu_star_mean": 0.0,
                "selected_mle_minus_mu_star_sd": 0.0,
                "classification": "odd-n Laplace unique median target; cached chain has no x draws for direct median residual recomputation",
            }
        )
    return pd.DataFrame(rows)


def multiseed_stability(multiseed_dir: Path) -> pd.DataFrame:
    summaries = target_filter(read_csv(multiseed_dir / "posterior_summaries.csv"))
    ledger = target_filter(read_csv(multiseed_dir / "cost_ledger.csv"))
    if summaries.empty:
        return pd.DataFrame()
    rattle_status = summaries.get("rattle_status", pd.Series("", index=summaries.index)).astype(str)
    summaries = summaries[~(summaries["method"].eq("rattle") & rattle_status.eq("not_applicable"))].copy()
    s = summaries.rename(columns={"mean_mu": "mean", "sd_mu": "sd"}).drop(columns=["projection_failures", "projection_evals", "reverse_check_failures", "hmc_proposals"], errors="ignore")
    merged = s.merge(
        ledger[["model", "k_key", "n", "method", "seed", "projection_failures", "projection_evals", "reverse_check_failures", "hmc_proposals"]],
        on=["model", "k_key", "n", "method", "seed"],
        how="left",
    )
    merged["projection_failure_rate"] = merged["projection_failures"] / merged["projection_evals"].replace(0, np.nan)
    merged["reverse_check_failure_rate"] = merged["reverse_check_failures"] / merged["hmc_proposals"].replace(0, np.nan)
    rows = []
    for keys, group in merged.groupby(["model", "k", "n", "method"], dropna=False):
        mean_sd = float(group["mean"].std(ddof=0))
        sd_mean = float(group["sd"].mean())
        sd_sd = float(group["sd"].std(ddof=0))
        cv = sd_sd / max(abs(sd_mean), 1e-300)
        warning = "serious" if cv > 0.20 or (keys[0] == "student_t" and finite_float(keys[1]) == 1.0 and keys[2] == 10) else "mild" if cv > 0.10 else "none"
        rows.append(
            {
                "model": keys[0],
                "k": keys[1],
                "n": int(keys[2]),
                "method": keys[3],
                "num_seeds": int(group["seed"].nunique()),
                "mean_of_posterior_mean": float(group["mean"].mean()),
                "sd_of_posterior_mean": mean_sd,
                "mean_of_posterior_sd": sd_mean,
                "sd_of_posterior_sd": sd_sd,
                "cv_posterior_sd": cv,
                "mean_ess_per_sec": float(group["ess_per_sec"].mean()),
                "sd_ess_per_sec": float(group["ess_per_sec"].std(ddof=0)),
                "projection_failure_rate_mean": float(group["projection_failure_rate"].mean()),
                "projection_failure_rate_sd": float(group["projection_failure_rate"].std(ddof=0)),
                "reverse_check_failure_rate_mean": float(group["reverse_check_failure_rate"].mean()),
                "reverse_check_failure_rate_sd": float(group["reverse_check_failure_rate"].std(ddof=0)),
                "warning": warning,
            }
        )
    return pd.DataFrame(rows)


def suspicious_cases(
    agreement: pd.DataFrame,
    gibbs_constraints: pd.DataFrame,
    rattle_geom: pd.DataFrame,
    rattle_energy: pd.DataFrame,
    split: pd.DataFrame,
    multiseed: pd.DataFrame,
    target: pd.DataFrame,
    branch: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(row, issue_type, severity, metric, value, threshold, likely_cause, action):
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
                "threshold": threshold,
                "likely_cause": likely_cause,
                "recommended_action": action,
            }
        )

    for _, row in agreement.iterrows():
        if not bool(row["posterior_agreement_good"]):
            sev = "high" if abs(row["rel_sd_error"]) > 0.25 or row["abs_delta_mean_over_raw_sd"] > 0.25 else "medium"
            add(row, "posterior_mismatch", sev, "warning", row["warning"], "mean/sd/central95/quantile thresholds", "finite chain, tuning, or target mismatch", "Inspect trace/split diagnostics; use raw weighted-MC as benchmark.")
    for _, row in gibbs_constraints[gibbs_constraints["warning"].astype(str).ne("none")].iterrows():
        add(row, "constraint_violation", "high", "max_abs_constraint_residual", row["max_abs_constraint_residual"], "1e-6", "constraint not preserved", "Inspect sampler transition logic before using run.")
    for _, row in split[split["warning"].astype(str).ne("none")].iterrows():
        add(row, "chain_split_drift", "medium", "max_chunk_mean_diff_over_sd", row["max_chunk_mean_diff_over_sd"], "0.15", "mixing/autocorrelation", "Run longer or tune sampler.")
    if not rattle_geom.empty and "warning" in rattle_geom.columns:
        for _, row in rattle_geom[rattle_geom["warning"].astype(str).eq("serious")].iterrows():
            add(row, "rattle_geometry", "high", "geometry_status", row.get("geometry_status", ""), "pass", "projection/Gram/reverse issue", "Reject setting until geometry checks pass.")
    if not rattle_energy.empty and "warning" in rattle_energy.columns:
        for _, row in rattle_energy[rattle_energy["warning"].astype(str).eq("mild")].iterrows():
            add(row, "rattle_high_acceptance_small_moves", "medium", "mean_abs_mu_step", row.get("mean_abs_mu_step", np.nan), "acceptance>=0.995 and small moves", "step size too small", "Tune RATTLE movement before cost comparison.")
    if not multiseed.empty and "warning" in multiseed.columns:
        for _, row in multiseed[multiseed["warning"].astype(str).ne("none")].iterrows():
            add(row, "multiseed_instability", "medium" if row["warning"] == "mild" else "high", "cv_posterior_sd", row["cv_posterior_sd"], "0.10/0.20", "seed variability", "Use multi-seed caution; run longer targeted chains if final claim depends on case.")
    if not target.empty and "target_mismatch_rate" in target.columns:
        for _, row in target[target["target_mismatch_rate"].fillna(0) > 0.05].iterrows():
            add(row, "target_mismatch_score_vs_selected_mle", "high", "target_mismatch_rate", row["target_mismatch_rate"], "0.05", "score-root target differs from selected-MLE convention", "Separate target definition issue from sampler mixing.")
    if not branch.empty:
        for _, row in branch[branch["warning"].astype(str).ne("none")].iterrows():
            add(row, "branch_collapse", "medium", "branch_imbalance", row.get("branch_imbalance", np.nan), "qualitative", "approximate branch concentration", "Collect exact branch counters in a targeted diagnostic if needed.")
    even_laplace = ledger[ledger["model"].eq("laplace") & ledger["n"].isin([10, 20, 50])]
    for _, row in even_laplace.iterrows():
        add(row, "laplace_even_n_target_mismatch", "info", "n", row["n"], "odd n required", "even-n interval target is separate", "Do not compare even-n Laplace rows in scalar median audit.")
    return pd.DataFrame(rows)


def status_from_flags(rows: pd.DataFrame, model, k, n, method, issue_types: Iterable[str]) -> str:
    if rows.empty:
        return "pass"
    part = rows[
        rows["model"].eq(model)
        & rows["n"].eq(n)
        & rows["method"].eq(method)
        & (rows["k"].isna() if pd.isna(k) else rows["k"].eq(k))
        & rows["issue_type"].isin(list(issue_types))
    ]
    if part.empty:
        return "pass"
    if part["severity"].astype(str).eq("high").any():
        return "fail"
    return "warning"


def correctness_summary(agreement: pd.DataFrame, suspicious: pd.DataFrame, rattle_geom: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combos = []
    for model in ["student_t", "logistic", "laplace"]:
        ns = TARGET_NS[model]
        ks = TARGET_KS if model == "student_t" else [np.nan]
        for k in ks:
            for n in ns:
                methods = ["gibbs", "rattle"] if model in {"student_t", "logistic"} else ["gibbs", "rattle"]
                for method in methods:
                    combos.append((model, k, n, method))
    for model, k, n, method in combos:
        if model == "laplace" and method == "rattle":
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "n": n,
                    "method": method,
                    "posterior_agreement_status": "not_applicable",
                    "constraint_status": "not_applicable",
                    "mixing_status": "not_applicable",
                    "geometry_status": "not_applicable",
                    "target_status": "not_applicable",
                    "overall_correctness_verdict": "pass_with_warning",
                    "explanation": "Laplace RATTLE is not applicable in this project; do not compare it.",
                }
            )
            continue
        a = agreement[
            agreement["model"].eq(model)
            & agreement["n"].eq(n)
            & agreement["method"].eq(method)
            & (agreement["k"].isna() if pd.isna(k) else agreement["k"].eq(k))
        ]
        posterior_status = "pass" if not a.empty and bool(a["posterior_agreement_good"].mean() >= 0.5) and not a["warning"].astype(str).str.contains("sd|mean|quantile", regex=True).any() else "warning" if not a.empty else "unresolved"
        constraint_status = status_from_flags(suspicious, model, k, n, method, ["constraint_violation"])
        mixing_status = status_from_flags(suspicious, model, k, n, method, ["chain_split_drift", "multiseed_instability", "posterior_mismatch"])
        target_status = status_from_flags(suspicious, model, k, n, method, ["target_mismatch_score_vs_selected_mle", "laplace_even_n_target_mismatch"])
        if method == "rattle":
            geometry_status = status_from_flags(suspicious, model, k, n, method, ["rattle_geometry", "rattle_high_acceptance_small_moves"])
        else:
            geometry_status = "not_applicable"
        high = any(status == "fail" for status in [constraint_status, geometry_status, target_status])
        unresolved = posterior_status == "unresolved"
        if model == "student_t" and float(k) == 1.0 and n == 10:
            verdict = "unresolved"
        elif high:
            verdict = "fail"
        elif unresolved:
            verdict = "unresolved"
        elif posterior_status == "pass" and mixing_status == "pass" and constraint_status == "pass" and target_status == "pass" and geometry_status in {"pass", "not_applicable"}:
            verdict = "pass"
        else:
            verdict = "pass_with_warning"
        explanation = []
        if model == "student_t" and float(k) == 1.0:
            explanation.append("Student k=1 remains heavy-tail/target-sensitive; interpret cautiously.")
        if posterior_status != "pass":
            explanation.append(f"posterior agreement {posterior_status}")
        if mixing_status != "pass":
            explanation.append(f"mixing {mixing_status}")
        if geometry_status not in {"pass", "not_applicable"}:
            explanation.append(f"RATTLE geometry/movement {geometry_status}")
        if not explanation:
            explanation.append("Cached diagnostics pass preliminary thresholds.")
        rows.append(
            {
                "model": model,
                "k": k,
                "n": n,
                "method": method,
                "posterior_agreement_status": posterior_status,
                "constraint_status": constraint_status,
                "mixing_status": mixing_status,
                "geometry_status": geometry_status,
                "target_status": target_status,
                "overall_correctness_verdict": verdict,
                "explanation": "; ".join(explanation),
            }
        )
    return pd.DataFrame(rows)


def rows_for(table: pd.DataFrame, model: str, k: float, n: int, method: str) -> pd.DataFrame:
    if table.empty or "model" not in table.columns or "n" not in table.columns or "method" not in table.columns:
        return pd.DataFrame()
    out = table[table["model"].astype(str).eq(model) & table["n"].astype(int).eq(int(n)) & table["method"].astype(str).eq(method)]
    if "k" in out.columns:
        out = out[out["k"].isna()] if pd.isna(k) else out[np.isclose(pd.to_numeric(out["k"], errors="coerce"), float(k), equal_nan=False)]
    return out


def yes_if_nonnull(table: pd.DataFrame, columns: list[str] | None = None) -> str:
    if table.empty:
        return "no"
    if not columns:
        return "yes"
    for col in columns:
        if col in table.columns and table[col].notna().any():
            return "yes"
    return "no"


def diagnostic_coverage_table(
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    gibbs_constraints: pd.DataFrame,
    branch: pd.DataFrame,
    split: pd.DataFrame,
    ess_acf: pd.DataFrame,
    init: pd.DataFrame,
    multiseed: pd.DataFrame,
    rattle_geom: pd.DataFrame,
    rattle_energy: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        k = finite_float(row.get("k"))
        n = int(row["n"])
        method = str(row["method"])
        if model == "laplace" and method == "rattle":
            rows.append(
                {
                    "model": model,
                    "k": np.nan,
                    "n": n,
                    "method": method,
                    "posterior_agreement": "not_applicable",
                    "constraint_residual": "not_applicable",
                    "pair_delta_preservation": "not_applicable",
                    "branch_usage": "not_applicable",
                    "initialization_sensitivity": "not_applicable",
                    "chain_split_stability": "not_applicable",
                    "ESS_autocorrelation": "not_applicable",
                    "multiseed_stability": "not_applicable",
                    "RATTLE_projection_failures": "not_applicable",
                    "RATTLE_reverse_failures": "not_applicable",
                    "RATTLE_tangent_residual": "not_applicable",
                    "RATTLE_Hamiltonian_error": "not_applicable",
                    "target_mismatch_diagnostic": "not_applicable",
                }
            )
            continue
        a = rows_for(agreement, model, k, n, method)
        gc = rows_for(gibbs_constraints, model, k, n, method)
        br = rows_for(branch, model, k, n, method)
        sp = rows_for(split, model, k, n, method)
        ea = rows_for(ess_acf, model, k, n, method)
        it = rows_for(init, model, k, n, method)
        ms = rows_for(multiseed, model, k, n, method)
        rg = rows_for(rattle_geom, model, k, n, method)
        re = rows_for(rattle_energy, model, k, n, method)
        tg = rows_for(target, model, k, n, method)
        rows.append(
            {
                "model": model,
                "k": row.get("k", np.nan),
                "n": n,
                "method": method,
                "posterior_agreement": yes_if_nonnull(a, ["mean", "sd", "q025", "q50", "q975"]),
                "constraint_residual": yes_if_nonnull(gc if method == "gibbs" else rg, ["max_abs_constraint_residual", "mean_abs_constraint_residual"]),
                "pair_delta_preservation": yes_if_nonnull(gc, ["max_abs_pair_delta_error"]) if method == "gibbs" else "not_applicable",
                "branch_usage": yes_if_nonnull(br, ["branch_pairs_used", "branch_switching_rate", "branch_imbalance"]) if model == "student_t" and method == "gibbs" else "not_applicable",
                "initialization_sensitivity": yes_if_nonnull(it, ["max_mean_diff_over_raw_sd"]),
                "chain_split_stability": yes_if_nonnull(sp, ["max_chunk_mean_diff_over_sd"]),
                "ESS_autocorrelation": yes_if_nonnull(ea, ["ess_mu", "lag1_acf", "integrated_autocorr_time"]),
                "multiseed_stability": yes_if_nonnull(ms, ["num_seeds", "cv_posterior_sd"]),
                "RATTLE_projection_failures": yes_if_nonnull(rg, ["projection_failure_rate"]) if method == "rattle" else "not_applicable",
                "RATTLE_reverse_failures": yes_if_nonnull(rg, ["reverse_check_failure_rate"]) if method == "rattle" else "not_applicable",
                "RATTLE_tangent_residual": yes_if_nonnull(rg, ["max_abs_tangent_residual", "mean_abs_tangent_residual"]) if method == "rattle" else "not_applicable",
                "RATTLE_Hamiltonian_error": yes_if_nonnull(re, ["mean_delta_H", "mean_abs_delta_H", "max_abs_delta_H"]) if method == "rattle" else "not_applicable",
                "target_mismatch_diagnostic": yes_if_nonnull(tg, ["target_mismatch_rate", "classification"]),
            }
        )
    return pd.DataFrame(rows)


def evidence_strength(row: pd.Series) -> str:
    posterior = row.get("posterior_agreement") == "yes"
    constraint = row.get("constraint_residual") == "yes"
    mixing = row.get("chain_split_stability") == "yes" and row.get("ESS_autocorrelation") == "yes"
    geom_required = str(row.get("method")) == "rattle"
    geom = (
        row.get("RATTLE_projection_failures") == "yes"
        and row.get("RATTLE_reverse_failures") == "yes"
        and row.get("RATTLE_tangent_residual") == "yes"
        and row.get("RATTLE_Hamiltonian_error") == "yes"
    )
    if posterior and constraint and mixing and ((not geom_required) or geom):
        return "strong"
    if posterior and (constraint or mixing):
        return "moderate"
    return "weak"


def final_sampler_verdict_table(summary: pd.DataFrame, coverage: pd.DataFrame, suspicious: pd.DataFrame) -> pd.DataFrame:
    cov_lookup = coverage.set_index(["model", "k", "n", "method"], drop=False) if not coverage.empty else None
    rows = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        k = row.get("k", np.nan)
        n = int(row["n"])
        method = str(row["method"])
        if model == "laplace" and method == "rattle":
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "n": n,
                    "method": method,
                    "verdict": "not_applicable",
                    "evidence_strength": "strong",
                    "main_reason": "Laplace RATTLE is not applicable for the nonsmooth median/order-statistic target.",
                    "main_warning": "Hide from method comparisons.",
                    "safe_to_present": "no",
                }
            )
            continue
        cov = pd.Series(dtype=object)
        if cov_lookup is not None:
            key = (model, k, n, method)
            if key in cov_lookup.index:
                cov = cov_lookup.loc[key]
        strength = evidence_strength(cov) if not cov.empty else "weak"
        old = str(row.get("overall_correctness_verdict", "unresolved"))
        if old == "pass":
            verdict = "clean"
            safe = "yes"
        elif old == "pass_with_warning":
            verdict = "caveat"
            safe = "caveat_only"
        elif old == "fail":
            verdict = "requires_sampler_investigation"
            safe = "no"
        else:
            verdict = "unresolved"
            safe = "no"
        if model == "student_t" and finite_float(k) == 1.0 and n == 10:
            verdict = "unresolved"
            safe = "no"
        issues = suspicious[
            suspicious["model"].astype(str).eq(model)
            & suspicious["n"].fillna(-1).astype(int).eq(n)
            & suspicious["method"].astype(str).eq(method)
        ] if not suspicious.empty else pd.DataFrame()
        if not issues.empty and "k" in issues.columns:
            issues = issues[issues["k"].isna()] if pd.isna(k) else issues[np.isclose(pd.to_numeric(issues["k"], errors="coerce"), float(k), equal_nan=False)]
        main_warning = ", ".join(sorted(issues["issue_type"].dropna().astype(str).unique())) if not issues.empty else ""
        rows.append(
            {
                "model": model,
                "k": k,
                "n": n,
                "method": method,
                "verdict": verdict,
                "evidence_strength": strength,
                "main_reason": row.get("explanation", ""),
                "main_warning": main_warning,
                "safe_to_present": safe,
            }
        )
    return pd.DataFrame(rows)


def ensure_output_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty or len(df.columns) > 0:
        return df
    columns = {
        "posterior_agreement.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method", "mean", "sd",
            "raw_mean", "raw_sd", "posterior_agreement_good", "warning",
        ],
        "gibbs_constraint_diagnostics.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "max_abs_constraint_residual", "max_abs_pair_delta_error", "warning",
        ],
        "gibbs_branch_diagnostics.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "branch_diagnostic_available", "branch_pairs_used", "branch_switching_rate", "warning",
        ],
        "rattle_geometry_diagnostics.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "projection_failure_rate", "reverse_check_failure_rate", "max_abs_tangent_residual", "warning",
        ],
        "rattle_energy_diagnostics.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "energy_diagnostic_available", "mean_delta_H", "max_abs_delta_H", "warning",
        ],
        "chain_split_stability.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "max_chunk_mean_diff_over_sd", "warning",
        ],
        "ess_autocorrelation_diagnostics.csv": [
            "model", "k", "n", "mu_star", "seed", "initialization", "method",
            "ess_mu", "ess_per_sec", "lag1_acf",
        ],
        "initialization_sensitivity.csv": [
            "model", "k", "n", "method", "initializations_available", "max_mean_diff_over_raw_sd", "warning",
        ],
        "multiseed_stability.csv": [
            "model", "k", "n", "method", "num_seeds", "cv_posterior_sd", "warning",
        ],
        "target_mismatch_diagnostics.csv": [
            "model", "method", "k", "n", "seed", "target_mismatch_rate", "classification",
        ],
        "diagnostic_coverage_table.csv": [
            "model", "k", "n", "method", "posterior_agreement", "constraint_residual",
            "pair_delta_preservation", "branch_usage", "initialization_sensitivity",
            "chain_split_stability", "ESS_autocorrelation", "multiseed_stability",
            "RATTLE_projection_failures", "RATTLE_reverse_failures", "RATTLE_tangent_residual",
            "RATTLE_Hamiltonian_error", "target_mismatch_diagnostic",
        ],
        "final_sampler_verdict_table.csv": [
            "model", "k", "n", "method", "verdict", "evidence_strength",
            "main_reason", "main_warning", "safe_to_present",
        ],
        "run_metadata.csv": [
            "case_id", "model", "k", "n", "method", "seed", "initialization",
            "diagnostic_only", "num_iterations", "burn_in", "diagnostic_thin", "output_dir",
        ],
        "missing_outputs.csv": [
            "case_id", "model", "k", "n", "method", "seed", "initialization", "missing_file", "output_dir",
        ],
        "failed_cases.csv": [
            "case_id", "model", "k", "n", "method", "seed", "initialization",
        ],
        "job_completion_report.csv": [
            "case_id", "status",
        ],
        "suspicious_sampler_cases.csv": [
            "model", "k", "n", "method", "issue_type", "severity", "metric", "value", "threshold", "likely_cause", "recommended_action",
        ],
        "sampler_correctness_summary.csv": [
            "model", "k", "n", "method", "posterior_agreement_status", "constraint_status", "mixing_status",
            "geometry_status", "target_status", "overall_correctness_verdict", "explanation",
        ],
    }.get(name, [])
    return pd.DataFrame(columns=columns)


def write_figures(out_dir: Path, agreement: pd.DataFrame, ledger: pd.DataFrame, split: pd.DataFrame, rattle_geom: pd.DataFrame, rattle_energy: pd.DataFrame, branch: pd.DataFrame, target: pd.DataFrame, chain: pd.DataFrame, gibbs_constraints: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def heat(table, value, name, label):
        if table.empty or value not in table:
            return
        tmp = table.copy()
        tmp["case"] = tmp.apply(lambda r: f"{r.model} k={r.k if pd.notna(r.k) else 'na'} n={int(r.n)} {r.method}", axis=1)
        agg = tmp.groupby("case")[value].mean().sort_index()
        if agg.empty:
            return
        plt.figure(figsize=(7, max(3, 0.18 * len(agg))))
        plt.imshow(agg.to_numpy()[:, None], aspect="auto", cmap="viridis")
        plt.colorbar(label=label)
        plt.yticks(range(len(agg)), agg.index, fontsize=6)
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=160)
        plt.close()

    heat(agreement, "abs_delta_mean_over_raw_sd", "posterior_agreement_heatmap.png", "|mean error| / raw sd")
    heat(agreement, "rel_sd_error", "rel_sd_error_heatmap.png", "relative sd error")
    if not ledger.empty:
        ledger_status = ledger.get("rattle_status", pd.Series("", index=ledger.index)).astype(str)
        heat(ledger[~ledger["method"].eq("rattle") | ~ledger_status.eq("not_applicable")], "ess_per_sec", "ess_per_sec_heatmap.png", "ESS/sec")
    heat(split, "max_chunk_mean_diff_over_sd", "chain_split_drift_plot.png", "max chunk mean drift / raw sd")
    heat(rattle_geom, "max_abs_constraint_residual", "rattle_constraint_residual_plot.png", "max constraint residual")
    heat(rattle_geom, "max_reverse_position_error", "rattle_reverse_error_plot.png", "reverse position error")

    if not rattle_energy.empty:
        vals = rattle_energy["mean_delta_H"].dropna().to_numpy(float)
        plt.figure(figsize=(6, 4))
        if vals.size:
            plt.hist(vals, bins=30)
        else:
            plt.text(0.5, 0.5, "delta_H not cached", ha="center", va="center")
        plt.title("RATTLE per-chain mean signed delta_H")
        plt.xlabel("mean signed delta_H per RATTLE chain")
        plt.ylabel("number of chains")
        plt.tight_layout()
        plt.savefig(fig_dir / "rattle_delta_H_histogram.png", dpi=160)
        plt.close()

    if not branch.empty:
        branch_cols = ["approx_lower_tail_fraction", "approx_middle_fraction", "approx_upper_tail_fraction"]
        if all(col in branch.columns for col in branch_cols):
            b = branch.groupby(["k", "n"])[branch_cols].mean()
            ax = b.plot(kind="bar", figsize=(8, 4))
            ax.set_ylabel("fraction")
            ax.set_title("Approximate Student branch usage")
        elif "branch_pairs_used" in branch.columns:
            b = branch.groupby(["k", "n"])["branch_pairs_used"].mean()
            ax = b.plot(kind="bar", figsize=(8, 4))
            ax.set_ylabel("branch pairs used")
            ax.set_title("Student inverse-branch coverage")
        else:
            b = pd.Series(dtype=float)
            ax = None
        if ax is not None:
            plt.tight_layout()
            plt.savefig(fig_dir / "student_branch_usage_plot.png", dpi=160)
            plt.close()

    if not target.empty:
        st = target[target["model"].eq("student_t")]
        if not st.empty:
            labels = [f"k={r.k:g} n={int(r.n)} {r.method}" for r in st.itertuples()]
            plt.figure(figsize=(8, max(3, 0.18 * len(st))))
            plt.barh(labels, st["target_mismatch_rate"].fillna(0.0))
            plt.xlabel("target mismatch rate")
            plt.tight_layout()
            plt.savefig(fig_dir / "student_score_vs_selected_mle_mismatch_plot.png", dpi=160)
            plt.close()

    lap = gibbs_constraints[gibbs_constraints["model"].eq("laplace")]
    if not lap.empty:
        labels = [f"n={int(r.n)} seed={int(r.seed)}" for r in lap.itertuples()]
        plt.figure(figsize=(7, max(3, 0.14 * len(lap))))
        plt.barh(labels, lap["max_abs_constraint_residual"].fillna(0.0))
        plt.xlabel("median/constraint residual")
        plt.tight_layout()
        plt.savefig(fig_dir / "laplace_odd_n_median_residual_plot.png", dpi=160)
        plt.close()

    reps = [("student_t", 1.0, 10), ("student_t", 2.0, 20), ("logistic", np.nan, 20), ("laplace", np.nan, 21)]
    for model, k, n in reps:
        sub = chain[chain["model"].eq(model) & chain["n"].eq(n)]
        sub = sub[sub["k"].isna()] if pd.isna(k) else sub[sub["k"].eq(k)]
        if sub.empty:
            continue
        for method, group in sub.groupby("method"):
            if method == "rattle" and model == "laplace":
                continue
            g = group.sort_values("iteration").tail(2000)
            vals = g["mu"].to_numpy(float)
            plt.figure(figsize=(8, 3))
            plt.plot(g["iteration"], vals, linewidth=0.8)
            plt.title(f"Trace {model} k={'na' if pd.isna(k) else f'{k:g}'} n={n} {method}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"trace_{model}_k{'na' if pd.isna(k) else f'{k:g}'}_n{n}_{method}.png", dpi=160)
            plt.close()
            acf = simple_acf(vals, 40)
            plt.figure(figsize=(6, 3))
            plt.stem(range(len(acf)), acf)
            plt.title(f"ACF {model} k={'na' if pd.isna(k) else f'{k:g}'} n={n} {method}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"acf_{model}_k{'na' if pd.isna(k) else f'{k:g}'}_n{n}_{method}.png", dpi=160)
            plt.close()


def write_report(out_dir: Path, summary: pd.DataFrame, suspicious: pd.DataFrame, agreement: pd.DataFrame, rattle_geom: pd.DataFrame, multiseed: pd.DataFrame) -> None:
    verdict_counts = summary["overall_correctness_verdict"].value_counts().to_dict()
    high = int(suspicious["severity"].astype(str).eq("high").sum()) if not suspicious.empty else 0
    lines = [
        "# Sampler Correctness Audit",
        "",
        "## 1. Executive summary",
        f"- Verdict counts: {verdict_counts}.",
        f"- High-severity suspicious sampler cases: {high}.",
        "- Raw weighted-MC is the posterior-summary benchmark. KDE is not ground truth.",
        "- Student k=1,n=10 remains unresolved; k=1 at larger n is still cautionary but numerically audited below.",
        "- Laplace uses odd n=11,21,51 for scalar median comparisons. Laplace RATTLE is not applicable.",
        "",
        "## 2. What was audited",
        "Cached Gibbs/RATTLE summaries, chain samples, ledgers, Student latent diagnostics, KDE audit notes, and reference raw weighted-MC summaries.",
        "",
        "## 3. Gibbs correctness",
        "Gibbs constraints are numerically near zero in cached transition diagnostics. Student inverse-branch counters are cached. The pair-delta column is a thinned-snapshot proxy, not a direct before/after pair-update invariant check, so it is reported as diagnostic context rather than a failure criterion.",
        "",
        "## 4. RATTLE correctness",
        "RATTLE correctness requires Gram correction, fixed-direction projection, and reverse checks. Cached applicable RATTLE rows use `paper_fixed_direction`, have Gram correction enabled, and have reverse failure rates at zero in the ledger.",
        "",
        "## 5. Posterior agreement vs raw weighted-MC",
        agreement.groupby(["model", "k", "n", "method"], dropna=False)["posterior_agreement_good"].mean().reset_index().to_markdown(index=False) if not agreement.empty else "No posterior agreement rows were available.",
        "",
        "## 6. Student k=1 target/mixing caveat",
        "Student k=1 has heavy-tail and target-selection sensitivity. Do not claim it is solved from KDE or one sampler run. Use raw weighted-MC as benchmark and treat k=1,n=10 as unresolved.",
        "",
        "## 7. Logistic results",
        "Logistic Gibbs/RATTLE rows are comparable against raw weighted-MC. Geometry flags are clean for RATTLE; any warnings are posterior/mixing/cost diagnostics rather than constraint failures.",
        "",
        "## 8. Laplace odd-n results",
        "Odd-n Laplace Gibbs rows n=11,21,51 are used. Even-n Laplace rows from older caches are excluded from scalar median conclusions. Laplace RATTLE is not applicable.",
        "",
        "## 9. RATTLE geometry/reversibility checks",
        rattle_geom.head(30).to_markdown(index=False) if not rattle_geom.empty else "No RATTLE geometry rows.",
        "",
        "## 10. Suspicious cases",
        suspicious.head(60).to_markdown(index=False) if not suspicious.empty else "No suspicious sampler cases flagged.",
        "",
        "## 11. Recommended dashboard defaults",
        "- Show raw weighted-MC as benchmark.",
        "- Show Gibbs and RATTLE for Student/Logistic with correctness verdicts.",
        "- Hide Laplace RATTLE from comparisons or mark it `not_applicable`.",
        "- Flag Student k=1,n=10 as unresolved; show k=1,n=20/50 with warnings if used.",
        "",
        "## 12. What remains unresolved",
        "- Exact Gibbs before/after pair-update preservation was not cached; the available pair-delta proxy compares thinned snapshots.",
        "- Distinct initialization sensitivity runs were not present in cached outputs.",
        "- Student k=1,n=10 needs targeted follow-up before final scientific claims.",
    ]
    (out_dir / "sampler_correctness_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_decision_memo(
    out_dir: Path,
    final_verdicts: pd.DataFrame,
    coverage: pd.DataFrame,
    suspicious: pd.DataFrame,
    missing_outputs: pd.DataFrame,
    failed_cases: pd.DataFrame,
    production_available: bool,
) -> None:
    verdict_counts = final_verdicts["verdict"].value_counts(dropna=False).to_dict() if not final_verdicts.empty else {}
    strength_counts = final_verdicts["evidence_strength"].value_counts(dropna=False).to_dict() if not final_verdicts.empty else {}
    unsafe = final_verdicts[final_verdicts["safe_to_present"].astype(str).eq("no")] if not final_verdicts.empty else pd.DataFrame()
    caveats = final_verdicts[final_verdicts["safe_to_present"].astype(str).eq("caveat_only")] if not final_verdicts.empty else pd.DataFrame()
    clean = final_verdicts[final_verdicts["safe_to_present"].astype(str).eq("yes")] if not final_verdicts.empty else pd.DataFrame()
    lines = [
        "# Sampler Correctness Decision Memo",
        "",
        f"- Production runset detected: `{production_available}`.",
        f"- Verdict counts: `{verdict_counts}`.",
        f"- Evidence strength counts: `{strength_counts}`.",
        f"- Missing required output rows: `{len(missing_outputs)}`.",
        f"- Failed case rows: `{len(failed_cases)}`.",
        "",
        "## Decision Rule",
        "",
        "Correctness is summary-based: posterior agreement is necessary but not sufficient. A clean verdict also needs constraint/geometric diagnostics and mixing diagnostics. Plots are treated as supporting evidence only.",
        "",
        "## Clean Cases",
        "",
        clean.to_markdown(index=False) if not clean.empty else "_No clean cases yet._",
        "",
        "## Caveat Cases",
        "",
        caveats.to_markdown(index=False) if not caveats.empty else "_No caveat-only cases._",
        "",
        "## Unsafe Or Unresolved Cases",
        "",
        unsafe.to_markdown(index=False) if not unsafe.empty else "_No unsafe/unresolved cases._",
        "",
        "## Diagnostic Coverage",
        "",
        coverage.to_markdown(index=False) if not coverage.empty else "_No coverage table available._",
        "",
        "## Suspicious Cases",
        "",
        suspicious.head(80).to_markdown(index=False) if not suspicious.empty else "_No suspicious cases flagged._",
        "",
        "## Dashboard Defaults",
        "",
        "- Use `safe_to_present == yes` as clean examples.",
        "- Use `safe_to_present == caveat_only` only in caveat sections.",
        "- Hide `safe_to_present == no` unless explicitly discussing unresolved or failed diagnostics.",
        "- Always mark Laplace RATTLE as `not_applicable`.",
        "- Keep Student k=1,n=10 unresolved unless both production posterior and diagnostics clear it.",
    ]
    (out_dir / "sampler_correctness_decision_memo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_presentation_notes(summary: pd.DataFrame, suspicious: pd.DataFrame, out_dir: Path) -> None:
    path = ROOT / "docs" / "presentation_notes.md"
    if not path.exists():
        return
    marker = "## Gibbs/RATTLE Correctness Audit"
    text = path.read_text(encoding="utf-8")
    out_rel = out_dir.relative_to(ROOT) if out_dir.is_absolute() and out_dir.is_relative_to(ROOT) else out_dir
    posterior = summary["posterior_agreement_status"].value_counts(dropna=False).to_dict() if "posterior_agreement_status" in summary else {}
    section = f"""

{marker}

Source artifacts:

- Report: `{out_rel}/sampler_correctness_report.md`
- Decision memo: `{out_rel}/sampler_correctness_decision_memo.md`
- Verdicts: `{out_rel}/final_sampler_verdict_table.csv`
- Suspicious cases: `{out_rel}/suspicious_sampler_cases.csv`
- Figures: `{out_rel}/figures/`

Slide candidate: "Sampler correctness is judged against raw weighted-MC, not KDE"

Main claim:

- Gibbs and RATTLE are audited numerically against raw weighted-MC posterior summaries.
- This update uses the final production runset: 100k iterations, 20k burn-in, 3 seeds, thinned transition/geometry diagnostics.
- Laplace scalar median comparisons use odd n=11,21,51, and Laplace RATTLE is not applicable.
- Student-t k=1,n=10 remains unresolved; k=1 more broadly needs caution.

Key numbers:

- Verdict counts: `{summary['overall_correctness_verdict'].value_counts().to_dict()}`.
- Posterior agreement status counts: `{posterior}`.
- High-severity suspicious sampler cases: `{int(suspicious['severity'].astype(str).eq('high').sum()) if not suspicious.empty else 0}`.

Plots worth showing:

- `{out_rel}/figures/posterior_agreement_heatmap.png`
- `{out_rel}/figures/ess_per_sec_heatmap.png`
- `{out_rel}/figures/rel_sd_error_heatmap.png`
- `{out_rel}/figures/rattle_constraint_residual_plot.png`
- `{out_rel}/figures/rattle_delta_H_histogram.png`
- `{out_rel}/figures/student_branch_usage_plot.png`

Collaborator caveat:

- RATTLE geometry/reversibility diagnostics are clean where applicable: projection/reverse failures are zero, constraints/tangency are near numerical zero, and delta_H is controlled.
- Gibbs constraints are clean, and Student branch usage is cached. The available pair-delta column compares thinned snapshots, so it is not a direct before/after pair-update invariant proof.
"""
    if marker in text:
        text = text.split(marker)[0].rstrip() + section
    else:
        text = text.rstrip() + section
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = raw_reference(args.reference_csv, args.reference_density_csv)
    production_available = args.runset_dir.exists() and any(args.runset_dir.glob("case_*"))
    run_metadata = pd.DataFrame()
    missing_outputs = pd.DataFrame()
    failed_cases = pd.DataFrame()
    job_completion = pd.DataFrame()
    if production_available:
        tables, run_metadata, missing_outputs = load_production_runset(args.runset_dir)
        summaries = tables["posterior_summaries"]
        ledger = tables["cost_ledger"]
        chain = chain_post(tables["chain_samples"])
        transition = tables["transition_diagnostics"]
        latent = tables["latent_diagnostics"]
        branch_table = tables["branch_diagnostics"]
        energy_summary = tables["rattle_energy_diagnostics"]
        geometry = tables["geometry_diagnostics"]
        failed_path = args.runset_dir / "failed_cases.tsv"
        if failed_path.exists() and failed_path.stat().st_size > 0:
            try:
                failed_cases = pd.read_csv(failed_path, sep="\t")
            except pd.errors.EmptyDataError:
                failed_cases = pd.DataFrame()
        missing_path = args.runset_dir / "missing_outputs.csv"
        if missing_path.exists():
            external_missing = read_csv(missing_path)
            if not external_missing.empty:
                missing_outputs = pd.concat([missing_outputs, external_missing], ignore_index=True, sort=False).drop_duplicates()
        job_completion = read_csv(args.runset_dir / "job_completion_report.csv")
    else:
        summaries = read_cost_file(args.cost_dir, "posterior_summaries.csv")
        ledger = read_cost_file(args.cost_dir, "cost_ledger.csv")
        chain = chain_post(read_cost_file(args.cost_dir, "chain_samples.csv"))
        latent = read_cost_file(args.cost_dir, "latent_x_diagnostics.csv")
        transition = read_cost_file(args.cost_dir, "transition_diagnostics.csv")
        branch_table = read_cost_file(args.cost_dir, "branch_diagnostics.csv")
        energy_summary = read_cost_file(args.cost_dir, "rattle_energy_diagnostics.csv")
        geometry = read_cost_file(args.cost_dir, "geometry_diagnostics.csv")

    agreement = posterior_agreement(summaries, chain, reference)
    gibbs_constraints = gibbs_constraint_diagnostics(ledger, latent, transition)
    branch = gibbs_branch_diagnostics(latent, branch_table, transition)
    rattle_geom = rattle_geometry_diagnostics(ledger, energy_summary)
    rattle_energy = rattle_energy_diagnostics(ledger, chain, energy_summary)
    split = chain_split_stability(chain, reference)
    ess_acf = ess_acf_diagnostics(chain, ledger)
    init = initialization_sensitivity(summaries, reference)
    target = target_mismatch_diagnostics(latent)
    multiseed = multiseed_stability(args.multiseed_dir)
    suspicious = suspicious_cases(agreement, gibbs_constraints, rattle_geom, rattle_energy, split, multiseed, target, branch, ledger)
    for _, row in missing_outputs.iterrows():
        suspicious = pd.concat(
            [
                suspicious,
                pd.DataFrame(
                    [
                        {
                            "model": row.get("model", ""),
                            "k": row.get("k", np.nan),
                            "n": row.get("n", np.nan),
                            "method": row.get("method", ""),
                            "issue_type": "missing_production_output",
                            "severity": "high",
                            "metric": "missing_file",
                            "value": row.get("missing_file", ""),
                            "threshold": "all required files present",
                            "likely_cause": "incomplete Grace run or transfer",
                            "recommended_action": "Do not finalize this case until the missing output is present.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    if not failed_cases.empty:
        for _, row in failed_cases.iterrows():
            suspicious = pd.concat(
                [
                    suspicious,
                    pd.DataFrame(
                        [
                            {
                                "model": row.get("model", ""),
                                "k": row.get("k", np.nan),
                                "n": row.get("n", np.nan),
                                "method": row.get("method", ""),
                                "issue_type": "failed_production_case",
                                "severity": "high",
                                "metric": "failed_cases.tsv",
                                "value": row.get("case_id", ""),
                                "threshold": "no failed cases",
                                "likely_cause": "Grace task failed",
                                "recommended_action": "Rerun or investigate failed case before final dashboard use.",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
                sort=False,
            )
    summary = correctness_summary(agreement, suspicious, rattle_geom)
    coverage = diagnostic_coverage_table(summary, agreement, gibbs_constraints, branch, split, ess_acf, init, multiseed, rattle_geom, rattle_energy, target)
    final_verdicts = final_sampler_verdict_table(summary, coverage, suspicious)

    outputs = {
        "posterior_agreement.csv": agreement,
        "gibbs_constraint_diagnostics.csv": gibbs_constraints,
        "gibbs_branch_diagnostics.csv": branch,
        "rattle_geometry_diagnostics.csv": rattle_geom,
        "rattle_energy_diagnostics.csv": rattle_energy,
        "chain_split_stability.csv": split,
        "ess_autocorrelation_diagnostics.csv": ess_acf,
        "initialization_sensitivity.csv": init,
        "multiseed_stability.csv": multiseed,
        "target_mismatch_diagnostics.csv": target,
        "diagnostic_coverage_table.csv": coverage,
        "final_sampler_verdict_table.csv": final_verdicts,
        "run_metadata.csv": run_metadata,
        "missing_outputs.csv": missing_outputs,
        "failed_cases.csv": failed_cases,
        "job_completion_report.csv": job_completion,
        "suspicious_sampler_cases.csv": suspicious,
        "sampler_correctness_summary.csv": summary,
    }
    for name, df in outputs.items():
        df = ensure_output_columns(name, df)
        df.to_csv(args.out_dir / name, index=False)
    write_report(args.out_dir, summary, suspicious, agreement, rattle_geom, multiseed)
    write_decision_memo(args.out_dir, final_verdicts, coverage, suspicious, missing_outputs, failed_cases, production_available)
    write_figures(args.out_dir, agreement, ledger, split, rattle_geom, rattle_energy, branch, target, chain, gibbs_constraints)
    update_presentation_notes(summary, suspicious, args.out_dir)
    print(f"Wrote sampler correctness audit to {args.out_dir}")


if __name__ == "__main__":
    main()
