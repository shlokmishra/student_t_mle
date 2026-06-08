"""Helpers for loading named analysis runsets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


COMMON_OUTPUTS = {
    "chain_samples": ("chain_samples.csv", "chain_samples.parquet"),
    "posterior_summaries": ("posterior_summaries.csv", "posterior_summaries.parquet"),
    "cost_ledger": ("cost_ledger.csv", "cost_ledger.parquet"),
    "diagnostic_summary": ("diagnostic_summary.csv", "diagnostic_summary.parquet"),
    "transition_diagnostics": ("transition_diagnostics.csv", "transition_diagnostics.parquet"),
    "latent_diagnostics": ("latent_diagnostics.csv", "latent_x_diagnostics.csv", "latent_diagnostics.parquet"),
    "geometry_diagnostics": ("geometry_diagnostics.csv", "geometry_diagnostics.parquet"),
    "rattle_energy_diagnostics": ("rattle_energy_diagnostics.csv", "rattle_energy_diagnostics.parquet"),
    "branch_diagnostics": ("branch_diagnostics.csv", "gibbs_branch_diagnostics.csv", "branch_diagnostics.parquet"),
    "initialization_diagnostics": ("initialization_diagnostics.csv", "initialization_diagnostics.parquet"),
}


@dataclass(frozen=True)
class Runset:
    name: str
    run_dir: Path
    reference_csv: Path | None
    label: str
    optional: bool = False


def load_run_registry(path: str | Path = "configs/analysis_run_registry.yaml") -> dict[str, dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Run registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Run registry must contain a mapping: {registry_path}")
    return data


def list_available_runsets(
    registry: dict[str, dict[str, Any]] | None = None,
    registry_path: str | Path = "configs/analysis_run_registry.yaml",
) -> list[str]:
    registry = load_run_registry(registry_path) if registry is None else registry
    return [name for name, cfg in registry.items() if Path(cfg.get("run_dir", "")).exists()]


def resolve_runset_paths(
    runset_name: str,
    registry: dict[str, dict[str, Any]] | None = None,
    registry_path: str | Path = "configs/analysis_run_registry.yaml",
) -> Runset:
    registry = load_run_registry(registry_path) if registry is None else registry
    if runset_name not in registry:
        raise KeyError(f"Unknown runset {runset_name!r}; available={sorted(registry)}")
    cfg = registry[runset_name] or {}
    run_dir = Path(cfg.get("run_dir", ""))
    ref = cfg.get("reference_csv")
    return Runset(
        name=runset_name,
        run_dir=run_dir,
        reference_csv=Path(ref) if ref else None,
        label=str(cfg.get("label", runset_name)),
        optional=bool(cfg.get("optional", False)),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _find_first(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _parse_case_id(case_dir: Path) -> dict[str, Any]:
    case_id = case_dir.name.removeprefix("case_")
    parts = case_id.split("_")
    meta: dict[str, Any] = {"case_id": case_id, "output_dir": str(case_dir)}
    try:
        if parts[:2] == ["student", "t"]:
            meta.update(
                {
                    "model": "student_t",
                    "k": float(parts[2].removeprefix("k")),
                    "n": int(parts[3].removeprefix("n")),
                    "method": parts[4],
                    "seed": int(parts[5].removeprefix("seed")),
                    "initialization": "_".join(parts[7:]) if len(parts) > 7 else "unspecified",
                }
            )
        elif parts and parts[0] in {"logistic", "laplace"}:
            meta.update(
                {
                    "model": parts[0],
                    "k": float("nan"),
                    "n": int(parts[1].removeprefix("n")),
                    "method": parts[2],
                    "seed": int(parts[3].removeprefix("seed")),
                    "initialization": "_".join(parts[5:]) if len(parts) > 5 else "unspecified",
                }
            )
    except (IndexError, ValueError):
        pass
    return meta


def _case_registry(directory: Path) -> dict[str, dict[str, Any]]:
    path = directory / "final_production_cases.tsv"
    if not path.exists():
        return {}
    try:
        cases = pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return {}
    out = {}
    for row in cases.to_dict(orient="records"):
        case_id = str(row.get("case_id", ""))
        if case_id:
            out[case_id] = row
            out[f"case_{case_id}"] = row
    return out


def _case_metadata(directory: Path, case_dir: Path, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = _parse_case_id(case_dir)
    meta.update(registry.get(meta["case_id"], registry.get(case_dir.name, {})))
    metadata_path = case_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta.update({key: value for key, value in loaded.items() if value is not None})
        except json.JSONDecodeError:
            meta["metadata_json_error"] = True
    meta.setdefault("case_id", case_dir.name.removeprefix("case_"))
    meta.setdefault("output_dir", str(case_dir))
    meta.setdefault("initialization", "unspecified")
    meta.setdefault("mu_star", 0.0)
    return meta


def _attach_metadata(frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
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
        "mu_star",
        "output_dir",
    ]:
        if col not in out.columns:
            out[col] = meta.get(col)
        else:
            out[col] = out[col].fillna(meta.get(col))
    if "iteration" in out.columns and "is_burn_in" not in out.columns and meta.get("burn_in") is not None:
        burn = pd.to_numeric(pd.Series([meta.get("burn_in")]), errors="coerce").iloc[0]
        if pd.notna(burn):
            out["is_burn_in"] = pd.to_numeric(out["iteration"], errors="coerce") < int(burn)
    return out


def _read_case_tables(directory: Path, names: tuple[str, ...]) -> tuple[pd.DataFrame, list[Path]]:
    frames = []
    paths = []
    registry = _case_registry(directory)
    for name in names:
        for path in sorted(directory.glob(f"case_*/{name}")):
            try:
                frame = _read_table(path)
            except Exception:
                continue
            if frame.empty:
                continue
            meta = _case_metadata(directory, path.parent, registry)
            frame = _attach_metadata(frame, meta)
            frame["case_dir"] = str(path.parent)
            frames.append(frame)
            paths.append(path)
        if frames:
            break
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True, sort=False), paths


def load_common_run_outputs(runset: Runset | dict[str, Any]) -> dict[str, Any]:
    if isinstance(runset, dict):
        runset = Runset(
            name=str(runset.get("name", runset.get("label", "runset"))),
            run_dir=Path(runset.get("run_dir", "")),
            reference_csv=Path(runset["reference_csv"]) if runset.get("reference_csv") else None,
            label=str(runset.get("label", runset.get("name", "runset"))),
            optional=bool(runset.get("optional", False)),
        )
    outputs: dict[str, Any] = {
        "runset": runset,
        "tables": {},
        "paths": {},
        "missing": [],
        "available": [],
    }
    if not runset.run_dir.exists():
        outputs["missing"].append(
            {
                "runset": runset.label,
                "diagnostic": "run_dir",
                "path": str(runset.run_dir),
                "severity": "info" if runset.optional else "high",
                "message": "Run directory is missing.",
            }
        )
        return outputs

    for key, names in COMMON_OUTPUTS.items():
        path = _find_first(runset.run_dir, names)
        if path is None:
            case_table, case_paths = _read_case_tables(runset.run_dir, names)
            if case_paths:
                outputs["tables"][key] = case_table
                outputs["paths"][key] = case_paths
                outputs["available"].append(
                    {
                        "runset": runset.label,
                        "diagnostic": key,
                        "path": f"{runset.run_dir}/case_*/{case_paths[0].name}",
                        "rows": int(len(case_table)),
                    }
                )
                continue
            outputs["tables"][key] = pd.DataFrame()
            outputs["missing"].append(
                {
                    "runset": runset.label,
                    "diagnostic": key,
                    "path": str(runset.run_dir),
                    "severity": "medium",
                    "message": "Common run output not found.",
                }
            )
            continue
        outputs["paths"][key] = path
        outputs["tables"][key] = _read_table(path)
        outputs["available"].append(
            {
                "runset": runset.label,
                "diagnostic": key,
                "path": str(path),
                "rows": int(len(outputs["tables"][key])),
            }
        )

    metadata_path = runset.run_dir / "run_metadata.json"
    if metadata_path.exists():
        outputs["paths"]["run_metadata"] = metadata_path
        outputs["available"].append(
            {"runset": runset.label, "diagnostic": "run_metadata", "path": str(metadata_path), "rows": 1}
        )
    else:
        outputs["missing"].append(
            {
                "runset": runset.label,
                "diagnostic": "run_metadata",
                "path": str(metadata_path),
                "severity": "low",
                "message": "Run metadata JSON not found.",
            }
        )
    return outputs
