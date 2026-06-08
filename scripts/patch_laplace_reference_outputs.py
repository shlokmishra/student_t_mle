"""Replace old Laplace reference rows with median-interval patch outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_N = {10, 20, 50}
EXPECTED_SEEDS = {123, 456, 789}
EXPECTED_CASES = len(EXPECTED_N) * len(EXPECTED_SEEDS)
EXPECTED_DENSITY_ROWS = EXPECTED_CASES * 2500
TARGET = "median_interval_contains_mu_star"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv"),
    )
    parser.add_argument(
        "--density-csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/reference_all_models_density_grid.csv"),
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("reporting/diagnostic_outputs/model_reference_audit/laplace_interval_patch"),
    )
    return parser.parse_args()


def read_parts(patch_dir: Path, pattern: str) -> pd.DataFrame:
    parts = sorted(patch_dir.glob(pattern))
    if len(parts) != EXPECTED_CASES:
        raise SystemExit(f"Expected {EXPECTED_CASES} files for {pattern}, found {len(parts)}")
    return pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)


def case_count(df: pd.DataFrame) -> int:
    return int(df[["n", "seed"]].drop_duplicates().shape[0])


def validate_summary(laplace: pd.DataFrame) -> None:
    if len(laplace) != EXPECTED_CASES:
        raise SystemExit(f"Expected {EXPECTED_CASES} final Laplace summary rows, found {len(laplace)}")
    if case_count(laplace) != EXPECTED_CASES:
        raise SystemExit(f"Expected {EXPECTED_CASES} unique Laplace n/seed cases, found {case_count(laplace)}")
    if set(laplace["n"].astype(int)) != EXPECTED_N:
        raise SystemExit("Final Laplace summary n values are not exactly 10,20,50")
    if set(laplace["seed"].astype(int)) != EXPECTED_SEEDS:
        raise SystemExit("Final Laplace summary seeds are not exactly 123,456,789")
    if set(laplace["estimator_type"].astype(str)) != {"raw_mc_interval_reference"}:
        raise SystemExit("Final Laplace summary estimator_type is not raw_mc_interval_reference")
    if set(laplace["backend"].astype(str)) != {"median_interval"}:
        raise SystemExit("Final Laplace summary backend is not median_interval")
    if set(laplace["target_description"].astype(str)) != {TARGET}:
        raise SystemExit(f"Final Laplace summary target_description is not {TARGET}")


def validate_density(laplace: pd.DataFrame) -> None:
    if len(laplace) != EXPECTED_DENSITY_ROWS:
        raise SystemExit(f"Expected {EXPECTED_DENSITY_ROWS} final Laplace density rows, found {len(laplace)}")
    if case_count(laplace) != EXPECTED_CASES:
        raise SystemExit(f"Expected {EXPECTED_CASES} unique Laplace density n/seed cases, found {case_count(laplace)}")
    if set(laplace["estimator_type"].astype(str)) != {"raw_mc_interval_reference"}:
        raise SystemExit("Final Laplace density estimator_type is not raw_mc_interval_reference")
    if set(laplace["backend"].astype(str)) != {"median_interval"}:
        raise SystemExit("Final Laplace density backend is not median_interval")
    if set(laplace["target_description"].astype(str)) != {TARGET}:
        raise SystemExit(f"Final Laplace density target_description is not {TARGET}")
    checks = pd.to_numeric(laplace["posterior_integral_check"], errors="coerce").dropna().to_numpy(dtype=float)
    if checks.size == 0 or not np.allclose(checks, 1.0, rtol=0.0, atol=1e-3):
        raise SystemExit("Final Laplace density posterior_integral_check is not close to 1")


def validate_no_old_laplace(summary: pd.DataFrame, density: pd.DataFrame) -> None:
    for name, df in [("summary", summary), ("density", density)]:
        laplace = df[df["model"].astype(str).eq("laplace")].copy()
        old = laplace[
            laplace["estimator_type"].astype(str).eq("kde_grid")
            & laplace["backend"].astype(str).isin(["scott", "SJ_transform", "t_abram"])
            & laplace["target_description"].astype(str).eq("deterministic_np_median_equals_mu_star")
        ]
        if not old.empty:
            raise SystemExit(f"Found old deterministic np.median KDE Laplace rows in final {name}: {len(old)}")


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    density = pd.read_csv(args.density_csv)
    patch_summary = read_parts(args.patch_dir, "part_laplace_n*_seed*.csv")
    patch_density = read_parts(args.patch_dir, "density_laplace_n*_seed*.csv")

    old_summary_rows = len(summary)
    old_density_rows = len(density)
    old_laplace_summary_rows = int(summary["model"].astype(str).eq("laplace").sum())
    old_laplace_density_rows = int(density["model"].astype(str).eq("laplace").sum())

    validate_summary(patch_summary)
    validate_density(patch_density)

    new_summary = pd.concat([summary[~summary["model"].astype(str).eq("laplace")], patch_summary], ignore_index=True)
    new_density = pd.concat([density[~density["model"].astype(str).eq("laplace")], patch_density], ignore_index=True)

    final_laplace_summary = new_summary[new_summary["model"].astype(str).eq("laplace")].copy()
    final_laplace_density = new_density[new_density["model"].astype(str).eq("laplace")].copy()
    validate_summary(final_laplace_summary)
    validate_density(final_laplace_density)
    validate_no_old_laplace(new_summary, new_density)

    new_summary.to_csv(args.summary_csv, index=False)
    new_density.to_csv(args.density_csv, index=False)

    print(f"summary rows: old={old_summary_rows} new={len(new_summary)}")
    print(f"summary Laplace rows: old={old_laplace_summary_rows} new={len(final_laplace_summary)}")
    print(f"density rows: old={old_density_rows} new={len(new_density)}")
    print(f"density Laplace rows: old={old_laplace_density_rows} new={len(final_laplace_density)}")
    print(f"final Laplace unique n/seed cases: {case_count(final_laplace_summary)}")
    print(f"final Laplace density posterior_integral_check min={final_laplace_density['posterior_integral_check'].min()} max={final_laplace_density['posterior_integral_check'].max()}")
    print("Laplace reference outputs patched successfully.")


if __name__ == "__main__":
    main()
