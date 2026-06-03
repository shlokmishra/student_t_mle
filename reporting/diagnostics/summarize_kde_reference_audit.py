"""Summarize KDE reference-audit CSVs.

Smoke run:
    python -m reporting.diagnostics.summarize_kde_reference_audit --csv reporting/diagnostic_outputs/kde_reference_audit/kde_reference_audit.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("reporting/diagnostic_outputs/kde_reference_audit/kde_reference_audit.csv"),
        help="CSV produced by audit_kde_reference.py.",
    )
    parser.add_argument("--stable-rel-tol", type=float, default=0.05, help="Heuristic relative variance tolerance for printed stability note.")
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _case_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["k"], row["n"], row["seed"], row["B"])


def _backend_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (*_case_key(row), row["backend"])


def print_raw(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    raw = {}
    print("\nRaw weighted-MC reference candidates")
    for row in rows:
        if row["estimator_type"] != "raw_weighted_mc":
            continue
        raw[_case_key(row)] = row
        print(
            "  "
            f"k={row['k']} n={row['n']} seed={row['seed']} B={row['B']} "
            f"mean={_f(row, 'posterior_mean'):.6g} sd={_f(row, 'posterior_sd'):.6g} "
            f"var={_f(row, 'posterior_var'):.6g} ess_w={_f(row, 'weighted_ess'):.1f}"
        )
    return raw


def print_grid_quad_diffs(rows: list[dict[str, Any]]) -> None:
    by_backend = defaultdict(lambda: {"grid": [], "quad": []})
    for row in rows:
        if row["estimator_type"] == "kde_grid":
            by_backend[_backend_key(row)]["grid"].append(row)
        elif row["estimator_type"] == "kde_quad":
            by_backend[_backend_key(row)]["quad"].append(row)

    print("\nKDE-grid vs KDE-quad differences")
    any_quad = False
    for key, vals in sorted(by_backend.items()):
        if not vals["quad"] or not vals["grid"]:
            continue
        any_quad = True
        quad = vals["quad"][0]
        qvar = _f(quad, "posterior_var")
        diffs = [abs(_f(g, "posterior_var") - qvar) / qvar for g in vals["grid"] if qvar > 0]
        print(
            "  "
            f"k={key[0]} n={key[1]} seed={key[2]} B={key[3]} backend={key[4]} "
            f"quad_var={qvar:.6g} grid_rel_diff_min={np.nanmin(diffs):.4g} "
            f"grid_rel_diff_max={np.nanmax(diffs):.4g}"
        )
    if not any_quad:
        print("  no kde_quad rows found; rerun audit with --use-quad to isolate integration-method effects.")


def print_grid_sensitivity(rows: list[dict[str, Any]]) -> None:
    grid_rows = [row for row in rows if row["estimator_type"] == "kde_grid"]
    by_bounds = defaultdict(list)
    by_ngrid = defaultdict(list)
    by_backend = defaultdict(list)
    for row in grid_rows:
        by_bounds[(*_backend_key(row), row["bound_multiplier"])].append(row)
        by_ngrid[(*_backend_key(row), row["n_grid"])].append(row)
        by_backend[_backend_key(row)].append(row)

    print("\nSensitivity to n_grid")
    for key, vals in sorted(by_bounds.items()):
        if len(vals) < 2:
            continue
        vars_ = np.asarray([_f(row, "posterior_var") for row in vals], dtype=float)
        rel_range = (np.nanmax(vars_) - np.nanmin(vars_)) / max(np.nanmean(vars_), 1e-300)
        print(f"  k={key[0]} n={key[1]} seed={key[2]} B={key[3]} backend={key[4]} bound={key[5]} rel_var_range={rel_range:.4g}")

    print("\nSensitivity to bounds")
    for key, vals in sorted(by_ngrid.items()):
        if len(vals) < 2:
            continue
        vars_ = np.asarray([_f(row, "posterior_var") for row in vals], dtype=float)
        rel_range = (np.nanmax(vars_) - np.nanmin(vars_)) / max(np.nanmean(vars_), 1e-300)
        print(f"  k={key[0]} n={key[1]} seed={key[2]} B={key[3]} backend={key[4]} n_grid={key[5]} rel_var_range={rel_range:.4g}")

    print("\nWithin-backend grid envelope")
    for key, vals in sorted(by_backend.items()):
        vars_ = np.asarray([_f(row, "posterior_var") for row in vals], dtype=float)
        if vars_.size == 0:
            continue
        rel_range = (np.nanmax(vars_) - np.nanmin(vars_)) / max(np.nanmean(vars_), 1e-300)
        print(f"  k={key[0]} n={key[1]} seed={key[2]} B={key[3]} backend={key[4]} rel_var_range={rel_range:.4g}")


def print_backend_sensitivity(rows: list[dict[str, Any]], raw: dict[tuple[str, str, str, str], dict[str, Any]], stable_rel_tol: float) -> None:
    grid_rows = [row for row in rows if row["estimator_type"] == "kde_grid"]
    by_case = defaultdict(list)
    for row in grid_rows:
        by_case[_case_key(row)].append(row)

    print("\nSensitivity to KDE backend/bandwidth and relative difference from raw weighted-MC")
    for key, vals in sorted(by_case.items()):
        # Use the largest grid and largest bound available for each backend to reduce grid effects.
        best_by_backend = {}
        for row in vals:
            backend = row["backend"]
            score = (_f(row, "bound_multiplier"), _f(row, "n_grid"))
            if backend not in best_by_backend or score > best_by_backend[backend][0]:
                best_by_backend[backend] = (score, row)
        selected = [item[1] for item in best_by_backend.values()]
        vars_ = np.asarray([_f(row, "posterior_var") for row in selected], dtype=float)
        rel_backend_range = (np.nanmax(vars_) - np.nanmin(vars_)) / max(np.nanmean(vars_), 1e-300)
        raw_var = _f(raw.get(key, {}), "posterior_var")
        print(f"  k={key[0]} n={key[1]} seed={key[2]} B={key[3]} backend_rel_var_range={rel_backend_range:.4g}")
        for row in sorted(selected, key=lambda item: item["backend"]):
            rel_raw = abs(_f(row, "posterior_var") - raw_var) / raw_var if raw_var > 0 else np.nan
            print(
                "    "
                f"{row['backend']}: var={_f(row, 'posterior_var'):.6g} sd={_f(row, 'posterior_sd'):.6g} "
                f"rel_var_diff_from_raw={rel_raw:.4g}"
            )
        stable_note = (
            "appears stable enough for this configured audit"
            if rel_backend_range <= stable_rel_tol
            else "shows material backend sensitivity in this configured audit"
        )
        print(f"    cautious note: posterior variance {stable_note}; inspect raw-MC ESS and grid/quad checks before using as a reference candidate.")


def main() -> None:
    args = parse_args()
    rows = _read(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    raw = print_raw(rows)
    print_grid_quad_diffs(rows)
    print_grid_sensitivity(rows)
    print_backend_sensitivity(rows, raw, args.stable_rel_tol)


if __name__ == "__main__":
    main()
