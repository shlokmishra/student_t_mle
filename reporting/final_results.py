"""Locked comparison results used for final reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

VALIDATION_PATH = Path("artifacts/final_comparison/final_validation_results.json")
KDE_AUDIT_PATH = Path("artifacts/final_comparison/kde_bandwidth_audit.json")


LOCKED_RESULTS: list[dict[str, Any]] = [
    {
        "model": "loc_logistic",
        "label": "Logistic",
        "baseline_family": "RATTLE",
        "baseline_status": "defensible",
        "baseline_reason": "Locked constrained HMC / RATTLE comparator passes reversibility and variance-agreement gates.",
        "baseline_config": {
            "step_size": 0.05,
            "num_steps": 2,
            "reverse_tol": 5e-3,
        },
        "reference_note": "notes/logistic_rattle_middle_regime.md",
        "runs": [
            {
                "n": 10,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.4144829246088024,
                        "posterior_var": 0.28896582409095906,
                        "ess_per_sec": 160.39210608865733,
                    },
                    "kde": {
                        "posterior_mean": 2.3933601876857358,
                        "posterior_var": 0.31358029012675503,
                    },
                    "rattle": {
                        "posterior_mean": 2.4122083432343624,
                        "posterior_var": 0.2990166071401459,
                        "runtime_s": 2.402786111831665,
                        "ess_per_sec": 219.3939830372197,
                        "x_accept": 0.9934,
                        "reverse_fail_rate": 0.0066,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.0066,
                        "projection_failures": 0,
                        "manifold_residual_mean": 6.0e-12,
                        "manifold_residual_max": 1.0e-10,
                    },
                    "full_data_mh": {
                        "posterior_mean": 2.402447565130097,
                        "posterior_var": 0.3042461756352176,
                        "ess_per_sec": 297.16870716390645,
                    },
                },
            },
            {
                "n": 20,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.286684455410812,
                        "posterior_var": 0.15241026904655658,
                        "ess_per_sec": 161.89503290607678,
                    },
                    "kde": {
                        "posterior_mean": 2.2880978933254488,
                        "posterior_var": 0.1537057968425844,
                    },
                    "rattle": {
                        "posterior_mean": 2.282301264167631,
                        "posterior_var": 0.14714710236376966,
                        "runtime_s": 2.3869894981384276,
                        "ess_per_sec": 214.23637685675322,
                        "x_accept": 0.9975333333333334,
                        "reverse_fail_rate": 0.0024666666666666665,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.0024666666666666665,
                        "projection_failures": 0,
                        "manifold_residual_mean": 6.0e-12,
                        "manifold_residual_max": 1.0e-10,
                    },
                    "full_data_mh": {
                        "posterior_mean": 2.2886553680023987,
                        "posterior_var": 0.14968320726819495,
                        "ess_per_sec": 340.9556507953636,
                    },
                },
            },
            {
                "n": 50,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.1564100753755584,
                        "posterior_var": 0.05855024152666943,
                        "ess_per_sec": 117.93196828911732,
                    },
                    "kde": {
                        "posterior_mean": 2.1533409079115655,
                        "posterior_var": 0.0614236323587233,
                    },
                    "rattle": {
                        "posterior_mean": 2.1578758923710164,
                        "posterior_var": 0.06074485490770033,
                        "runtime_s": 2.442802619934082,
                        "ess_per_sec": 207.04070668105754,
                        "x_accept": 0.9992000000000001,
                        "reverse_fail_rate": 0.0008,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.0008,
                        "projection_failures": 0,
                        "manifold_residual_mean": 6.0e-12,
                        "manifold_residual_max": 1.0e-10,
                    },
                    "full_data_mh": {
                        "posterior_mean": 2.1553624863443384,
                        "posterior_var": 0.05929942248626023,
                        "ess_per_sec": 265.383009133062,
                    },
                },
            },
        ],
    },
    {
        "model": "loc_student_k3",
        "label": "Student-3",
        "baseline_family": "RATTLE",
        "baseline_status": "defensible",
        "baseline_reason": "Frozen Student-3 comparator is stable on the n-grid with modest tolerance-only reverse failures.",
        "baseline_config": {
            "step_size": 0.05,
            "num_steps": 1,
            "reverse_tol": 1e-2,
        },
        "reference_note": "notes/student3_rattle_grid.md",
        "runs": [
            {
                "n": 10,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 1.7690727795439216,
                        "posterior_var": 0.17034556755840122,
                        "ess_per_sec": 149.53375267317625,
                    },
                    "kde": {
                        "posterior_mean": 1.7578370045458542,
                        "posterior_var": 0.16289587985570003,
                    },
                    "rattle": {
                        "posterior_mean": 1.758709573467287,
                        "posterior_var": 0.17036307788434488,
                        "runtime_s": 2.2288188457489015,
                        "ess_per_sec": 231.91308807738105,
                        "x_accept": 0.9799,
                        "reverse_fail_rate": 0.020066666666666667,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.020066666666666667,
                        "projection_failures": 0,
                        "manifold_residual_mean": 2.68e-12,
                        "manifold_residual_max": 9.94e-11,
                    },
                    "full_data_mh": {
                        "ess_per_sec": 344.42137362791857,
                    },
                },
            },
            {
                "n": 20,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.0285065136407967,
                        "posterior_var": 0.08265670607869828,
                        "ess_per_sec": 140.66854719245674,
                    },
                    "kde": {
                        "posterior_mean": 2.030787792651044,
                        "posterior_var": 0.07954735860946187,
                    },
                    "rattle": {
                        "posterior_mean": 2.0286397932734666,
                        "posterior_var": 0.07520772087129945,
                        "runtime_s": 2.241848516464233,
                        "ess_per_sec": 233.95370814399536,
                        "x_accept": 0.9863,
                        "reverse_fail_rate": 0.013599999999999998,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.013599999999999998,
                        "projection_failures": 0,
                        "manifold_residual_mean": 2.12e-12,
                        "manifold_residual_max": 9.83e-11,
                    },
                    "full_data_mh": {
                        "ess_per_sec": 338.84359091081643,
                    },
                },
            },
            {
                "n": 50,
                "seed_count": 5,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.0691023048384443,
                        "posterior_var": 0.029684260283837093,
                        "ess_per_sec": 123.33766539841909,
                    },
                    "kde": {
                        "posterior_mean": 2.0717982835118502,
                        "posterior_var": 0.031405083494819255,
                    },
                    "rattle": {
                        "posterior_mean": 2.0704465327149273,
                        "posterior_var": 0.027838921218494645,
                        "runtime_s": 2.255189800262451,
                        "ess_per_sec": 160.31324218632676,
                        "x_accept": 0.9922,
                        "reverse_fail_rate": 0.007733333333333333,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 0.0,
                        "reverse_tolerance_only_rate": 0.007733333333333333,
                        "projection_failures": 0,
                        "manifold_residual_mean": 2.76e-12,
                        "manifold_residual_max": 9.90e-11,
                    },
                    "full_data_mh": {
                        "ess_per_sec": 245.858897799404,
                    },
                },
            },
        ],
    },
    {
        "model": "loc_student_k2",
        "label": "Student-2",
        "baseline_family": "RATTLE",
        "baseline_status": "not defensible",
        "baseline_reason": "Projection and reversibility are manageable, but the baseline remains under-dispersed relative to Gibbs/KDE across the narrow calibration window.",
        "baseline_config": {
            "step_size": 0.05,
            "num_steps": 1,
            "reverse_tol": 1e-2,
        },
        "reference_note": "notes/student2_rattle_status.md",
        "runs": [
            {
                "n": 20,
                "seed_count": 1,
                "methods": {
                    "gibbs": {
                        "posterior_mean": 2.404900480829099,
                        "posterior_var": 0.09246536869333301,
                        "runtime_s": 5.3151,
                        "ess": 572.6,
                        "ess_per_sec": 107.7,
                    },
                    "kde": {
                        "posterior_mean": 2.391402193735771,
                        "posterior_var": 0.09078959597355607,
                        "runtime_s": 2.9242,
                    },
                    "rattle": {
                        "posterior_mean": 2.381995498192003,
                        "posterior_var": 0.0717677647178695,
                        "runtime_s": 2.5917,
                        "ess": 597.0,
                        "ess_per_sec": 230.33977025232662,
                        "x_accept": 0.9547,
                        "reverse_fail_rate": 136.0 / 3000.0,
                        "reverse_projection_fail_rate": 0.0,
                        "reverse_position_mismatch_rate": 0.0,
                        "reverse_momentum_mismatch_rate": 1.0 / 3000.0,
                        "reverse_tolerance_only_rate": 135.0 / 3000.0,
                        "projection_failures": 0,
                        "manifold_residual_mean": 1.65e-12,
                        "manifold_residual_max": 9.95e-11,
                    },
                    "full_data_mh": {
                        "posterior_mean": 2.4067,
                        "posterior_var": 0.0918,
                        "runtime_s": 1.7793,
                        "ess": 534.1,
                        "ess_per_sec": 300.1,
                    },
                },
            },
        ],
        "calibration_shortlist": [
            {
                "step_size": 0.02,
                "num_steps": 1,
                "reverse_tol": 1e-2,
                "reverse_fail_rate": 0.0,
                "posterior_var": 0.0554,
                "kde_var": 0.0933,
                "gibbs_var": 0.0922,
                "ess_per_sec": 215.7,
            },
            {
                "step_size": 0.03,
                "num_steps": 1,
                "reverse_tol": 1e-2,
                "reverse_fail_rate": 0.0009,
                "posterior_var": 0.0683,
                "kde_var": 0.0933,
                "gibbs_var": 0.0922,
                "ess_per_sec": 250.0,
            },
            {
                "step_size": 0.03,
                "num_steps": 2,
                "reverse_tol": 2e-2,
                "reverse_fail_rate": 0.0011,
                "posterior_var": 0.0783,
                "kde_var": 0.0933,
                "gibbs_var": 0.0922,
                "ess_per_sec": 202.9,
            },
            {
                "step_size": 0.05,
                "num_steps": 1,
                "reverse_tol": 2e-2,
                "reverse_fail_rate": 0.0061,
                "posterior_var": 0.0765,
                "kde_var": 0.0933,
                "gibbs_var": 0.0922,
                "ess_per_sec": 215.5,
            },
            {
                "step_size": 0.04,
                "num_steps": 2,
                "reverse_tol": 2e-2,
                "reverse_fail_rate": 0.0190,
                "posterior_var": 0.0836,
                "kde_var": 0.0933,
                "gibbs_var": 0.0922,
                "ess_per_sec": 205.1,
            },
        ],
    },
    {
        "model": "loc_laplace",
        "label": "Laplace",
        "baseline_family": "RATTLE",
        "baseline_status": "not attempted",
        "baseline_reason": "Final package keeps Laplace as a Gibbs/KDE-only comparison because the main baseline story is already decided by Logistic and Student tails.",
        "baseline_config": None,
        "reference_note": None,
        "runs": [],
    },
    {
        "model": "loc_cauchy",
        "label": "Cauchy",
        "baseline_family": "RATTLE",
        "baseline_status": "not attempted",
        "baseline_reason": "Cauchy RATTLE was intentionally not pursued after Student-2 already exposed heavy-tail fragility.",
        "baseline_config": None,
        "reference_note": None,
        "runs": [],
    },
]


def _flatten_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_entry in LOCKED_RESULTS:
        for run in model_entry.get("runs", []):
            for method, metrics in run.get("methods", {}).items():
                row = {
                    "experiment_group": "locked",
                    "experiment_label": "locked_summary",
                    "model": model_entry["model"],
                    "label": model_entry["label"],
                    "n": run["n"],
                    "seed_count": run.get("seed_count"),
                    "method": method,
                    "baseline_status": model_entry["baseline_status"],
                    "reference_note": model_entry.get("reference_note"),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def _load_validation_payload() -> dict[str, Any] | None:
    if not VALIDATION_PATH.exists():
        return None
    with VALIDATION_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_kde_audit_payload() -> dict[str, Any] | None:
    if not KDE_AUDIT_PATH.exists():
        return None
    with KDE_AUDIT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_validation_rows(validation_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not validation_payload:
        return []

    rows: list[dict[str, Any]] = []
    student2 = validation_payload.get("student2_replication", {})
    for config in student2.get("configs", []):
        for method, metrics in config.get("aggregate", {}).get("methods", {}).items():
            row = {
                "experiment_group": "validation_student2_replication",
                "experiment_label": config["name"],
                "model": "loc_student_k2",
                "label": "Student-2",
                "n": config.get("n"),
                "seed_count": config.get("seed_count"),
                "method": method,
                "baseline_status": "not defensible",
                "reference_note": "notes/student2_rattle_status.md",
            }
            row.update(metrics)
            rows.append(row)

    for check in validation_payload.get("long_chain_checks", []):
        model_name = "loc_logistic" if check["model"] == "loc_logistic" else f"loc_student_k{int(check['k'])}"
        label = "Logistic" if check["model"] == "loc_logistic" else f"Student-{int(check['k'])}"
        for method, metrics in check.get("result", {}).get("methods", {}).items():
            row = {
                "experiment_group": "validation_long_chain",
                "experiment_label": check["name"],
                "model": model_name,
                "label": label,
                "n": check.get("n"),
                "seed_count": 1,
                "method": method,
                "baseline_status": "defensible",
                "reference_note": "notes/final_validation.md",
            }
            row.update(metrics)
            rows.append(row)

    return rows


def _build_method_status_rows(
    validation_payload: dict[str, Any] | None,
    kde_audit_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    student2_lookup = {}
    long_chain_lookup = {}
    audit_lookup = {}
    if validation_payload:
        for cfg in validation_payload.get("student2_replication", {}).get("configs", []):
            student2_lookup[cfg["name"]] = cfg
        for check in validation_payload.get("long_chain_checks", []):
            if check["model"] == "loc_logistic":
                long_chain_lookup["loc_logistic"] = check
            elif check["model"] == "loc_student" and float(check["k"]) == 3.0:
                long_chain_lookup["loc_student_k3"] = check
    if kde_audit_payload:
        audit_lookup = {entry["model"]: entry for entry in kde_audit_payload.get("model_audits", [])}

    for model_entry in LOCKED_RESULTS:
        representative_run = None
        for run in model_entry.get("runs", []):
            if run.get("n") == 20:
                representative_run = run
                break
        if representative_run is None and model_entry.get("runs"):
            representative_run = model_entry["runs"][0]

        baseline_metrics = {}
        if representative_run is not None:
            baseline_metrics = representative_run.get("methods", {}).get("rattle", {})
            kde_metrics = representative_run.get("methods", {}).get("kde", {})
            if baseline_metrics and kde_metrics:
                kde_var = kde_metrics.get("posterior_var")
                base_var = baseline_metrics.get("posterior_var")
                if kde_var:
                    baseline_metrics = {
                        **baseline_metrics,
                        "variance_rel_err_vs_kde": abs(base_var - kde_var) / kde_var,
                    }

        validation_summary = None
        if model_entry["model"] == "loc_student_k2":
            frozen = student2_lookup.get("frozen_student3_transfer")
            closest = student2_lookup.get("closest_shortlist_candidate")
            if frozen and closest:
                validation_summary = (
                    "Student-2 under-dispersion replicated across 3 seeds for both the frozen transfer "
                    f"({100.0 * frozen['aggregate']['underdispersion_vs_kde']:.1f}% below KDE variance) "
                    "and the closest shortlist candidate "
                    f"({100.0 * closest['aggregate']['underdispersion_vs_kde']:.1f}% below KDE variance)."
                )
        elif model_entry["model"] in long_chain_lookup:
            check = long_chain_lookup[model_entry["model"]]
            m = check["result"]["methods"]
            validation_summary = (
                f"Long-chain check at n={check['n']} kept RATTLE ESS/sec above Gibbs "
                f"({m['rattle']['ess_per_sec']:.1f} vs {m['gibbs']['ess_per_sec']:.1f}) "
                f"with reverse fail {100.0 * m['rattle']['reverse_fail_rate']:.2f}%."
            )
        audit_entry = audit_lookup.get(model_entry["model"])

        rows.append(
            {
                "model": model_entry["model"],
                "label": model_entry["label"],
                "baseline_status": model_entry["baseline_status"],
                "baseline_reason": model_entry["baseline_reason"],
                "baseline_config": model_entry.get("baseline_config"),
                "representative_n": representative_run.get("n") if representative_run else None,
                "reverse_fail_rate": baseline_metrics.get("reverse_fail_rate"),
                "reverse_projection_fail_rate": baseline_metrics.get("reverse_projection_fail_rate"),
                "reverse_position_mismatch_rate": baseline_metrics.get("reverse_position_mismatch_rate"),
                "reverse_momentum_mismatch_rate": baseline_metrics.get("reverse_momentum_mismatch_rate"),
                "reverse_tolerance_only_rate": baseline_metrics.get("reverse_tolerance_only_rate"),
                "projection_failures": baseline_metrics.get("projection_failures"),
                "variance_rel_err_vs_kde": baseline_metrics.get("variance_rel_err_vs_kde"),
                "ess_per_sec": baseline_metrics.get("ess_per_sec"),
                "runtime_s": baseline_metrics.get("runtime_s"),
                "validation_summary": validation_summary,
                "package_ready_to_share": validation_payload.get("package_ready_to_share") if validation_payload else None,
                "kde_backend_recommendation": audit_entry.get("recommended", {}).get("name") if audit_entry else None,
                "kde_backend_reason": (
                    "held-out MLE log score plus posterior normalization and tail sanity checks"
                    if audit_entry else None
                ),
            }
        )
    return rows


def build_results_payload() -> dict[str, Any]:
    validation_payload = _load_validation_payload()
    kde_audit_payload = _load_kde_audit_payload()
    return {
        "models": LOCKED_RESULTS,
        "validation_checks": validation_payload,
        "kde_bandwidth_audit": kde_audit_payload,
        "summary_rows": _flatten_summary_rows() + _flatten_validation_rows(validation_payload),
        "method_status_rows": _build_method_status_rows(validation_payload, kde_audit_payload),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_outputs(out_dir: str | Path) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = build_results_payload()
    json_path = out_path / "final_results.json"
    summary_csv_path = out_path / "final_summary_rows.csv"
    status_csv_path = out_path / "method_status_rows.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(summary_csv_path, payload["summary_rows"])
    _write_csv(status_csv_path, payload["method_status_rows"])

    return {
        "json": json_path,
        "summary_csv": summary_csv_path,
        "status_csv": status_csv_path,
    }


if __name__ == "__main__":
    paths = write_summary_outputs("artifacts/final_comparison")
    for key, value in paths.items():
        print(f"{key}: {value}")
