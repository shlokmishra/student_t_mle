"""Map the n=3 Student score manifold and test RATTLE component communication.

The score coordinates are ``z_i = y_i / (k + y_i**2)`` with
``z_1 + z_2 + z_3 = 0``.  Each non-boundary score value has an inner and an
outer inverse, so the n=3 manifold is represented by eight branch sheets over
the two-dimensional feasible hexagon.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.loc.student import score_gibbs as loc_student
from models.loc.student import score_rattle as loc_student_rattle
from models.loc.student.mle import loglik
from reporting.diagnostics.study_student_mle_transitions import mle_competitor_margin


def _ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=_ints, default=[1, 2, 3])
    parser.add_argument("--seeds", type=_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gibbs-iterations", type=int, default=12_000)
    parser.add_argument("--gibbs-burn-in", type=int, default=2_000)
    parser.add_argument("--rattle-iterations", type=int, default=8_000)
    parser.add_argument("--rattle-burn-in", type=int, default=1_000)
    parser.add_argument("--restarts-per-sheet", type=int, default=3)
    parser.add_argument("--map-resolution", type=int, default=81)
    parser.add_argument("--proposal-std-mu", type=float, default=1.1)
    parser.add_argument("--proposal-std-z", type=float, default=0.16)
    parser.add_argument("--rattle-step-size", type=float, default=0.10)
    parser.add_argument("--rattle-num-steps", type=int, default=5)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def score_coordinates(x: np.ndarray, mu_star: float, k: float) -> np.ndarray:
    y = np.asarray(x, dtype=float) - mu_star
    return y / (k + y * y)


def branch_bits(x: np.ndarray, mu_star: float, k: float) -> tuple[int, int, int]:
    y = np.asarray(x, dtype=float) - mu_star
    return tuple(int(value) for value in (np.abs(y) > math.sqrt(k)))


def branch_code(x: np.ndarray, mu_star: float, k: float) -> str:
    return "".join(str(value) for value in branch_bits(x, mu_star, k))


def inverse_score(z: float, outer: int, k: float) -> float:
    """Stable finite inverse of z=y/(k+y^2); outer=1 selects |y|>sqrt(k)."""
    zmax = 0.5 / math.sqrt(k)
    if abs(z) > zmax * (1.0 + 1e-12):
        raise ValueError("score coordinate outside its feasible interval")
    discriminant = max(0.0, 1.0 - 4.0 * k * z * z)
    root = math.sqrt(discriminant)
    if outer:
        if z == 0.0:
            return math.copysign(math.inf, z)
        return (1.0 + root) / (2.0 * z)
    return 2.0 * k * z / (1.0 + root)


def x_from_z_branch(z: np.ndarray, bits: tuple[int, int, int], k: float, mu_star: float) -> np.ndarray:
    return mu_star + np.asarray(
        [inverse_score(float(value), bit, k) for value, bit in zip(z, bits)],
        dtype=float,
    )


def classify(x: np.ndarray, mu_star: float, k: float) -> dict:
    return mle_competitor_margin(x, mu_star, k)


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def map_branch_sheets(args: argparse.Namespace) -> list[dict]:
    """Map normalized score geometry once; it is invariant to k rescaling."""
    k = 1.0
    values = np.linspace(-1.0, 1.0, args.map_resolution)
    rows = []
    for u1 in values:
        for u2 in values:
            u3 = -u1 - u2
            if abs(u3) > 1.0 + 1e-12:
                continue
            z = 0.5 * np.asarray([u1, u2, u3])
            for integer_code in range(8):
                bits = tuple((integer_code >> shift) & 1 for shift in (2, 1, 0))
                if any(bit and abs(value) < 1e-12 for bit, value in zip(bits, z)):
                    continue
                x = x_from_z_branch(z, bits, k, args.mu_star)
                if not np.all(np.isfinite(x)):
                    continue
                result = classify(x, args.mu_star, k)
                rows.append(
                    {
                        "branch": "".join(str(bit) for bit in bits),
                        "u1": float(u1),
                        "u2": float(u2),
                        "u3": float(u3),
                        "mle_indicator": int(result["mu_star_is_global_mle"]),
                        "num_stationary_roots": result["num_stationary_points"],
                        "num_local_maxima": result["num_local_maxima"],
                        "mle_margin": result["mle_margin"],
                    }
                )
    return rows


def sampler_params(args: argparse.Namespace, k: float, iterations: int) -> dict:
    return {
        "k": float(k),
        "n": 3,
        "num_iterations_T": int(iterations),
        "proposal_std_mu": args.proposal_std_mu,
        "proposal_std_z": args.proposal_std_z,
        "prior_mean": 0.0,
        "prior_std": 10.0,
        "gibbs_backend": "numba_full",
        "initialization": "central",
        "store_x_chain": True,
        "rattle_step_size": args.rattle_step_size,
        "rattle_num_steps": args.rattle_num_steps,
    }


def classify_chain(states: np.ndarray, mu_star: float, k: float) -> tuple[list[str], np.ndarray, list[dict]]:
    codes = [branch_code(x, mu_star, k) for x in states]
    details = [classify(x, mu_star, k) for x in states]
    indicators = np.asarray([item["mu_star_is_global_mle"] for item in details], dtype=bool)
    return codes, indicators, details


def transition_rows(
    *,
    method: str,
    k: float,
    chain_id: str,
    initial_kind: str,
    states: np.ndarray,
    codes: list[str],
    indicators: np.ndarray,
    mu_star: float,
) -> tuple[list[dict], list[dict]]:
    counts = Counter(zip(codes[:-1], codes[1:]))
    matrix_rows = []
    outgoing = Counter(codes[:-1])
    for (before, after), count in sorted(counts.items()):
        matrix_rows.append(
            {
                "method": method,
                "k": k,
                "chain_id": chain_id,
                "initial_kind": initial_kind,
                "branch_before": before,
                "branch_after": after,
                "count": count,
                "probability": count / outgoing[before],
            }
        )

    zmax = 0.5 / math.sqrt(k)
    switch_rows = []
    for index, (before, after) in enumerate(zip(codes[:-1], codes[1:])):
        if before == after:
            continue
        changed = [i for i in range(3) if before[i] != after[i]]
        z_before = score_coordinates(states[index], mu_star, k)
        z_after = score_coordinates(states[index + 1], mu_star, k)
        endpoint_gaps = [
            min(abs(abs(z_before[i]) - zmax), abs(abs(z_after[i]) - zmax)) / zmax
            for i in changed
        ]
        switch_rows.append(
            {
                "method": method,
                "k": k,
                "chain_id": chain_id,
                "initial_kind": initial_kind,
                "iteration": index,
                "branch_before": before,
                "branch_after": after,
                "coordinates_changed": len(changed),
                "min_normalized_endpoint_merge_gap": min(endpoint_gaps),
                "max_normalized_endpoint_merge_gap": max(endpoint_gaps),
                "mle_exit": int(indicators[index] and not indicators[index + 1]),
                "mle_return": int(not indicators[index] and indicators[index + 1]),
            }
        )
    return matrix_rows, switch_rows


def choose_restart_states(
    candidates: dict[str, list[np.ndarray]], restarts_per_sheet: int
) -> list[tuple[str, int, np.ndarray]]:
    chosen = []
    for code, states in sorted(candidates.items()):
        if not states:
            continue
        indices = np.linspace(0, len(states) - 1, min(restarts_per_sheet, len(states)), dtype=int)
        for restart_index, state_index in enumerate(np.unique(indices)):
            chosen.append((code, restart_index, states[int(state_index)].copy()))
    return chosen


def chain_summary(
    *,
    method: str,
    k: float,
    chain_id: str,
    initial_kind: str,
    initial_branch: str,
    states: np.ndarray,
    codes: list[str],
    indicators: np.ndarray,
    details: list[dict],
    diagnostics: dict | None,
) -> dict:
    exits = int(np.sum(indicators[:-1] & ~indicators[1:]))
    returns = int(np.sum(~indicators[:-1] & indicators[1:]))
    initial_mle = bool(indicators[0])
    if initial_mle:
        outcome = "crossed_both_ways" if exits and returns else ("exited_mle" if exits else "remained_mle")
    elif not np.any(indicators):
        outcome = "remained_non_mle"
    elif exits:
        outcome = "returned_then_crossed_both_ways"
    else:
        outcome = "returned_to_mle_only"
    return {
        "method": method,
        "k": k,
        "chain_id": chain_id,
        "initial_kind": initial_kind,
        "initial_branch": initial_branch,
        "initial_mle_indicator": int(initial_mle),
        "states": len(states),
        "outside_occupancy": float(np.mean(~indicators)),
        "mle_exit_count": exits,
        "mle_return_count": returns,
        "branches_visited": " ".join(sorted(set(codes))),
        "num_branches_visited": len(set(codes)),
        "minimum_mle_margin": float(np.min([item["mle_margin"] for item in details])),
        "maximum_mle_margin": float(np.max([item["mle_margin"] for item in details])),
        "outcome": outcome,
        "x_acceptance_rate": float(diagnostics.get("x_acceptance_rate", math.nan)) if diagnostics else math.nan,
        "projection_failures": int(diagnostics.get("projection_failure_count", 0)) if diagnostics else 0,
        "reverse_failures": int(diagnostics.get("reverse_check_failure_count", 0)) if diagnostics else 0,
    }


def likelihood_curve_rows(
    k: float,
    method: str,
    event: str,
    event_id: str,
    before: np.ndarray,
    after: np.ndarray,
    mu_star: float,
) -> list[dict]:
    lower = min(float(np.min(before)), float(np.min(after)), mu_star) - 2.0 * math.sqrt(k)
    upper = max(float(np.max(before)), float(np.max(after)), mu_star) + 2.0 * math.sqrt(k)
    grid = np.linspace(lower, upper, 401)
    rows = []
    for state_label, x in (("before", before), ("after", after)):
        values = np.asarray([loglik(x, mu, k) for mu in grid])
        values -= np.max(values)
        for mu, value in zip(grid, values):
            rows.append(
                {
                    "k": k,
                    "method": method,
                    "event": event,
                    "event_id": event_id,
                    "state": state_label,
                    "mu": float(mu),
                    "relative_loglik": float(value),
                }
            )
    return rows


def plot_likelihood_curves(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    for k in sorted({row["k"] for row in rows}):
        selected_k = [row for row in rows if row["k"] == k]
        events = list(dict.fromkeys(row["event_id"] for row in selected_k))
        fig, axes = plt.subplots(len(events), 1, figsize=(8, 2.8 * len(events)), squeeze=False)
        for axis, event_id in zip(axes[:, 0], events):
            selected = [row for row in selected_k if row["event_id"] == event_id]
            for state, style in (("before", "-"), ("after", "--")):
                subset = [row for row in selected if row["state"] == state]
                axis.plot(
                    [row["mu"] for row in subset],
                    [row["relative_loglik"] for row in subset],
                    style,
                    label=state,
                )
            axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
            axis.set(title=event_id, xlabel=r"$\mu$", ylabel="relative log likelihood")
            axis.legend()
        fig.tight_layout()
        fig.savefig(out.with_name(f"{out.stem}_k{int(k)}{out.suffix}"), dpi=180)
        plt.close(fig)


def plot_branch_sheet_maps(rows: list[dict], out: Path) -> None:
    """Plot MLE membership and finite competing-maximum margins on all sheets."""
    branches = sorted({str(row["branch"]).zfill(3) for row in rows})
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for axis, branch in zip(axes.flat, branches):
        selected = [row for row in rows if str(row["branch"]).zfill(3) == branch]
        colors = [row["mle_indicator"] for row in selected]
        axis.scatter(
            [row["u1"] for row in selected],
            [row["u2"] for row in selected],
            c=colors,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=5,
            linewidths=0,
        )
        axis.set_title(f"branch {branch}")
        axis.set_aspect("equal")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$u_1=2\sqrt{k}z_1$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$u_2=2\sqrt{k}z_2$")
    fig.suptitle("n=3 branch sheets: red = MLE-valid, blue = non-MLE")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    specifications = [
        ("num_stationary_roots", "number of stationary roots", "viridis", BoundaryNorm(np.arange(0.5, 6.5), 256)),
        ("num_local_maxima", "number of local maxima", "viridis", BoundaryNorm(np.arange(0.5, 4.5), 256)),
    ]
    for field, title, cmap, norm in specifications:
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
        for axis, branch in zip(axes.flat, branches):
            selected = [row for row in rows if str(row["branch"]).zfill(3) == branch]
            scatter = axis.scatter(
                [row["u1"] for row in selected], [row["u2"] for row in selected],
                c=[row[field] for row in selected], cmap=cmap, norm=norm,
                s=5, linewidths=0,
            )
            axis.set_title(f"branch {branch}")
            axis.set_aspect("equal")
        fig.colorbar(scatter, ax=axes, shrink=0.8, label=title)
        fig.suptitle(f"n=3 branch sheets: {title}")
        fig.savefig(out.with_name(f"branch_sheet_{field}.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    finite_margins = np.asarray(
        [row["mle_margin"] for row in rows if math.isfinite(row["mle_margin"])], dtype=float
    )
    cap = float(np.quantile(np.abs(finite_margins), 0.99))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    for axis, branch in zip(axes.flat, branches):
        selected = [row for row in rows if str(row["branch"]).zfill(3) == branch]
        margins = [
            cap if not math.isfinite(row["mle_margin"]) else float(np.clip(row["mle_margin"], -cap, cap))
            for row in selected
        ]
        scatter = axis.scatter(
            [row["u1"] for row in selected], [row["u2"] for row in selected],
            c=margins, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap),
            s=5, linewidths=0,
        )
        axis.set_title(f"branch {branch}")
        axis.set_aspect("equal")
    fig.colorbar(scatter, ax=axes, shrink=0.8, label=r"$D(x)$ (clipped; no competitor at upper cap)")
    fig.suptitle("n=3 branch sheets: competing-maximum likelihood margin")
    fig.savefig(out.with_name("branch_sheet_mle_margin.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def switch_direction_summary(switches: list[dict]) -> list[dict]:
    counts: Counter = Counter()
    for row in switches:
        before = str(row["branch_before"])
        after = str(row["branch_after"])
        for old, new in zip(before, after):
            if old == new:
                continue
            direction = "inner_to_outer" if (old, new) == ("0", "1") else "outer_to_inner"
            key = (
                row["method"], row["k"], row["initial_kind"], direction,
                row["mle_exit"], row["mle_return"],
            )
            counts[key] += 1
    return [
        {
            "method": key[0],
            "k": key[1],
            "initial_kind": key[2],
            "direction": key[3],
            "mle_exit": key[4],
            "mle_return": key[5],
            "coordinate_switch_count": count,
        }
        for key, count in sorted(counts.items())
    ]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sheet_rows = map_branch_sheets(args)
    _write(args.out / "branch_sheet_map.csv", sheet_rows)

    summaries: list[dict] = []
    matrices: list[dict] = []
    switches: list[dict] = []
    restart_rows: list[dict] = []
    curve_rows: list[dict] = []

    for k_integer in args.k_values:
        k = float(k_integer)
        non_mle_candidates: dict[str, list[np.ndarray]] = defaultdict(list)
        mle_candidates: dict[str, list[np.ndarray]] = defaultdict(list)
        for seed in args.seeds:
            params = sampler_params(args, k, args.gibbs_iterations)
            chain = loc_student.run_gibbs(np.array([seed, 3], dtype=np.uint32), args.mu_star, params, verbose=False)
            states = np.asarray(chain["x_chain"], dtype=float)[args.gibbs_burn_in :]
            codes, indicators, details = classify_chain(states, args.mu_star, k)
            chain_id = f"gibbs-k{k_integer}-seed{seed}"
            summary = chain_summary(
                method="gibbs", k=k, chain_id=chain_id, initial_kind="central",
                initial_branch=codes[0], states=states, codes=codes,
                indicators=indicators, details=details, diagnostics=None,
            )
            summaries.append(summary)
            matrix, switch = transition_rows(
                method="gibbs", k=k, chain_id=chain_id, initial_kind="central",
                states=states, codes=codes, indicators=indicators, mu_star=args.mu_star,
            )
            matrices.extend(matrix)
            switches.extend(switch)
            for x, code, indicator in zip(states, codes, indicators):
                target = mle_candidates if indicator else non_mle_candidates
                if len(target[code]) < 200:
                    target[code].append(x.copy())
            exit_indices = np.flatnonzero(indicators[:-1] & ~indicators[1:])[:1]
            for index in exit_indices:
                curve_rows.extend(
                    likelihood_curve_rows(
                        k, "gibbs", "mle_exit", f"Gibbs k={k_integer}, seed={seed} exit",
                        states[index], states[index + 1], args.mu_star,
                    )
                )

        restarts = choose_restart_states(non_mle_candidates, args.restarts_per_sheet)
        # Add one MLE start from each sheet not represented among the non-MLE starts.
        restart_codes = {code for code, _, _ in restarts}
        for code, _, x in choose_restart_states(mle_candidates, 1):
            if code not in restart_codes:
                restarts.append((code, 0, x))

        for restart_number, (source_code, within_sheet, initial_x) in enumerate(restarts):
            initial_class = classify(initial_x, args.mu_star, k)
            initial_kind = "gibbs_non_mle_restart" if not initial_class["mu_star_is_global_mle"] else "gibbs_mle_restart"
            params = sampler_params(args, k, args.rattle_iterations)
            params["initial_x"] = initial_x
            key = np.array([10_000 + 100 * k_integer + restart_number, 3], dtype=np.uint32)
            chain = loc_student_rattle.run_rattle(key, args.mu_star, params, verbose=False)
            states = np.asarray(chain["x_chain"], dtype=float)[args.rattle_burn_in :]
            codes, indicators, details = classify_chain(states, args.mu_star, k)
            chain_id = f"rattle-k{k_integer}-{source_code}-{within_sheet}-{restart_number}"
            diagnostics = chain["projection_diagnostics"]
            summary = chain_summary(
                method="rattle", k=k, chain_id=chain_id, initial_kind=initial_kind,
                initial_branch=source_code, states=states, codes=codes,
                indicators=indicators, details=details, diagnostics=diagnostics,
            )
            summaries.append(summary)
            restart_rows.append(
                {
                    **summary,
                    "source_x1": float(initial_x[0]),
                    "source_x2": float(initial_x[1]),
                    "source_x3": float(initial_x[2]),
                    "source_mle_margin": float(initial_class["mle_margin"]),
                }
            )
            matrix, switch = transition_rows(
                method="rattle", k=k, chain_id=chain_id, initial_kind=initial_kind,
                states=states, codes=codes, indicators=indicators, mu_star=args.mu_star,
            )
            matrices.extend(matrix)
            switches.extend(switch)
            crossings = np.flatnonzero(indicators[:-1] != indicators[1:])[:1]
            for index in crossings:
                event = "mle_exit" if indicators[index] else "mle_return"
                curve_rows.extend(
                    likelihood_curve_rows(
                        k, "rattle", event, f"RATTLE k={k_integer} {event}",
                        states[index], states[index + 1], args.mu_star,
                    )
                )
            print(
                f"k={k_integer} restart {source_code}/{within_sheet}: {summary['outcome']}; "
                f"visited {summary['branches_visited']}",
                flush=True,
            )

    _write(args.out / "chain_summary.csv", summaries)
    _write(args.out / "rattle_restart_summary.csv", restart_rows)
    _write(args.out / "branch_transition_matrix.csv", matrices)
    _write(args.out / "branch_switch_events.csv", switches)
    _write(args.out / "branch_switch_direction_summary.csv", switch_direction_summary(switches))
    _write(args.out / "likelihood_curves.csv", curve_rows)
    plot_likelihood_curves(curve_rows, args.out / "likelihood_curves.png")
    plot_branch_sheet_maps(sheet_rows, args.out / "branch_sheet_mle_map.png")
    print(f"wrote results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
