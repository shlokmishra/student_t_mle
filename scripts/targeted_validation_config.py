"""Tiny config loader for targeted validation YAML-like files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_case_config(path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    section = None
    current: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line == "defaults:":
            section = "defaults"
            current = None
            continue
        if line == "cases:":
            section = "cases"
            current = None
            continue
        stripped = line.strip()
        if section == "defaults" and ":" in stripped:
            key, value = stripped.split(":", 1)
            defaults[key.strip()] = parse_scalar(value)
            continue
        if section == "cases":
            if stripped.startswith("- "):
                current = {}
                cases.append(current)
                stripped = stripped[2:]
            if current is not None and ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = parse_scalar(value)
                continue
        raise ValueError(f"Could not parse config line: {raw}")
    return {"defaults": defaults, "cases": cases}


def expanded_cases(path: Path) -> list[dict[str, Any]]:
    config = load_case_config(path)
    defaults = config["defaults"]
    expanded: list[dict[str, Any]] = []
    for case in config["cases"]:
        seeds = case.get("seeds", defaults.get("seeds", [0]))
        inits = case.get("initializations", defaults.get("initializations", ["central"]))
        for seed in seeds:
            for init in inits:
                row = {
                    **defaults,
                    **case,
                    "seed": int(seed),
                    "initialization": str(init),
                    "num_iterations": int(case.get("num_iterations", defaults.get("num_iterations", 30000))),
                    "burn_in": int(case.get("burn_in", defaults.get("burn_in", 5000))),
                    "diagnostic_thin": int(case.get("diagnostic_thin", defaults.get("diagnostic_thin", 20))),
                    "diagnostic_only": bool(case.get("diagnostic_only", False)),
                }
                base = str(case["case_id"])
                row["case_id"] = f"{base}_seed{seed}_init_{init}"
                expanded.append(row)
    return expanded


def find_case(path: Path, case_id: str) -> dict[str, Any]:
    matches = [case for case in expanded_cases(path) if str(case["case_id"]) == str(case_id)]
    if not matches:
        available = ", ".join(case["case_id"] for case in expanded_cases(path)[:10])
        raise KeyError(f"Unknown case_id={case_id}. First available cases: {available}")
    return matches[0]
