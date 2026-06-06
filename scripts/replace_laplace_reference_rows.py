"""Replace Laplace rows in a reference CSV with freshly generated part CSVs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--new-parts-glob", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--model", default="laplace")
    return parser.parse_args()


def read_parts(pattern: str) -> pd.DataFrame:
    parts = [Path(path) for path in sorted(glob.glob(pattern))]
    if not parts:
        raise FileNotFoundError(f"No replacement part CSVs matched: {pattern}")
    frames = [pd.read_csv(path) for path in parts]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    if not args.main_csv.exists():
        raise FileNotFoundError(f"Existing main CSV not found: {args.main_csv}")

    current = pd.read_csv(args.main_csv)
    replacement = read_parts(args.new_parts_glob)
    if "model" not in current.columns or "model" not in replacement.columns:
        raise ValueError("Both current and replacement CSVs must contain a 'model' column.")

    model = str(args.model)
    kept = current[~current["model"].astype(str).eq(model)].copy()
    fresh = replacement[replacement["model"].astype(str).eq(model)].copy()
    if fresh.empty:
        raise ValueError(f"Replacement CSVs contain no rows for model={model!r}.")

    missing_in_fresh = [column for column in current.columns if column not in fresh.columns]
    for column in missing_in_fresh:
        fresh[column] = pd.NA
    extra_in_fresh = [column for column in fresh.columns if column not in current.columns]
    if extra_in_fresh:
        kept = kept.reindex(columns=list(current.columns) + extra_in_fresh)
    output_columns = list(kept.columns)
    fresh = fresh.reindex(columns=output_columns)
    out = pd.concat([kept, fresh], ignore_index=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(
        f"replaced model={model}: kept_rows={len(kept)} "
        f"replacement_rows={len(fresh)} output_rows={len(out)} out={args.out_csv}"
    )


if __name__ == "__main__":
    main()
