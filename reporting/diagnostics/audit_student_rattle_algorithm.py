"""Local mechanics audit for the Student-t RATTLE implementation.

This is not a sampler profiler and does not run long chains.  It checks the
equation-level properties that should hold before interpreting timing curves:

* the Gram-corrected potential gradient matches finite differences;
* projected random states remain on the score manifold;
* one forward RATTLE trajectory followed by the reverse trajectory returns to
  the original position;
* proposed momenta remain tangent;
* energy error grows as step size and dimension make integration harder.

Example:
    python -m reporting.diagnostics.audit_student_rattle_algorithm \
        --n-values 10,50,200,1000 --eps-values 0.005,0.02,0.05,0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import loc_student_rattle as rattle


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--mu-star", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.2)
    parser.add_argument("--n-values", type=_ints, default=[10, 50, 200, 1000])
    parser.add_argument("--eps-values", type=_floats, default=[0.005, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seeds", type=_ints, default=list(range(20)))
    parser.add_argument("--out", type=Path, default=Path("results/student_rattle_algorithm_audit"))
    parser.add_argument("--proj-tol", type=float, default=1e-10)
    parser.add_argument("--proj-max-iters", type=int, default=25)
    parser.add_argument("--grad-tol", type=float, default=1e-12)
    parser.add_argument("--tangent-tol", type=float, default=1e-8)
    return parser.parse_args()


def _finite_difference_grad(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for idx in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (fun(x_plus) - fun(x_minus)) / (2.0 * eps)
    return grad


def _project_state(x: np.ndarray, mu_star: float, k: float) -> tuple[np.ndarray, bool, float]:
    x_proj, ok, _, residual = rattle._project_to_manifold_newton(
        x,
        mu_star,
        k,
        tol=1e-12,
        max_iters=50,
        grad_tol=1e-14,
        damping=1.0,
        line_search=True,
        init_strategy="trial",
    )
    return x_proj, bool(ok), float(residual)


def _project_momentum(x: np.ndarray, momentum: np.ndarray, mu_star: float, k: float) -> tuple[np.ndarray, bool]:
    grad = rattle._constraint_grad(x, mu_star, k)
    return rattle._project_momentum(momentum, grad, grad_tol=1e-14)


def gradient_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(101)
    for n in args.n_values:
        # Keep finite differences cheap; this is a mechanics check, not a timing run.
        n_fd = min(int(n), 50)
        x = rng.normal(size=n_fd)
        analytic = rattle._grad_potential(
            x,
            args.mu,
            args.k,
            mu_star=args.mu_star,
            include_gram_correction=True,
        )
        numeric = _finite_difference_grad(
            lambda z: rattle._potential_energy(
                z,
                args.mu,
                args.k,
                mu_star=args.mu_star,
                include_gram_correction=True,
            ),
            x,
        )
        max_abs = float(np.max(np.abs(analytic - numeric)))
        rel = max_abs / max(1.0, float(np.max(np.abs(numeric))))
        rows.append(
            {
                "check": "gram_corrected_gradient_finite_difference",
                "n_requested": int(n),
                "n_checked": int(n_fd),
                "max_abs_error": max_abs,
                "relative_error": float(rel),
                "passed": bool(max_abs <= 1e-6),
            }
        )
    return rows


def trajectory_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    total = len(args.n_values) * len(args.eps_values) * len(args.seeds)
    case_idx = 0
    for eps in args.eps_values:
        for n in args.n_values:
            for seed in args.seeds:
                case_idx += 1
                print(
                    f"[{case_idx}/{total}] RATTLE mechanics n={n} eps={eps:g} seed={seed}",
                    flush=True,
                )
                rng = np.random.default_rng(seed)
                x0, init_ok, init_residual = _project_state(rng.normal(size=n), args.mu_star, args.k)
                if not init_ok:
                    rows.append(
                        {
                            "n": int(n),
                            "eps": float(eps),
                            "seed": int(seed),
                            "stage": "initial_projection",
                            "ok": False,
                            "initial_constraint_residual": init_residual,
                        }
                    )
                    continue
                p0, p_ok = _project_momentum(x0, rng.normal(size=n), args.mu_star, args.k)
                if not p_ok:
                    rows.append({"n": int(n), "eps": float(eps), "seed": int(seed), "stage": "initial_momentum", "ok": False})
                    continue

                h0 = rattle._hamiltonian(
                    x0,
                    p0,
                    args.mu,
                    args.k,
                    mu_star=args.mu_star,
                    include_gram_correction=True,
                )
                x1, p1, ok_forward, diag = rattle._rattle_trajectory(
                    x0,
                    p0,
                    args.mu,
                    args.mu_star,
                    args.k,
                    step_size=eps,
                    num_steps=args.num_steps,
                    proj_tol=args.proj_tol,
                    proj_max_iters=args.proj_max_iters,
                    grad_tol=args.grad_tol,
                    proj_damping=1.0,
                    proj_line_search=True,
                    proj_init_strategy="trial",
                    include_gram_correction=True,
                    projection_mode="paper_fixed_direction",
                    tangent_tol=args.tangent_tol,
                )
                if not ok_forward:
                    rows.append(
                        {
                            "n": int(n),
                            "eps": float(eps),
                            "seed": int(seed),
                            "stage": "forward",
                            "ok": False,
                            "forward_projection_iterations": int(diag["projection_iterations"]),
                        }
                    )
                    continue

                h1 = rattle._hamiltonian(
                    x1,
                    p1,
                    args.mu,
                    args.k,
                    mu_star=args.mu_star,
                    include_gram_correction=True,
                )
                xr, pr, ok_reverse, rev_diag = rattle._rattle_trajectory(
                    x1,
                    -p1,
                    args.mu,
                    args.mu_star,
                    args.k,
                    step_size=eps,
                    num_steps=args.num_steps,
                    proj_tol=args.proj_tol,
                    proj_max_iters=args.proj_max_iters,
                    grad_tol=args.grad_tol,
                    proj_damping=1.0,
                    proj_line_search=True,
                    proj_init_strategy="trial",
                    include_gram_correction=True,
                    projection_mode="paper_fixed_direction",
                    tangent_tol=args.tangent_tol,
                )
                grad1 = rattle._constraint_grad(x1, args.mu_star, args.k)
                rows.append(
                    {
                        "n": int(n),
                        "eps": float(eps),
                        "seed": int(seed),
                        "stage": "completed",
                        "ok": bool(ok_reverse),
                        "initial_constraint_residual": init_residual,
                        "proposal_constraint_residual": abs(rattle._constraint_value(x1, args.mu_star, args.k)),
                        "proposal_tangent_residual": abs(float(np.dot(grad1, p1))),
                        "reverse_position_error": float(np.linalg.norm(xr - x0)),
                        "reverse_momentum_error": float(np.linalg.norm(pr + p0)),
                        "delta_h": float(h1 - h0),
                        "abs_delta_h": abs(float(h1 - h0)),
                        "forward_projection_iterations": int(diag["projection_iterations"]),
                        "reverse_projection_iterations": int(rev_diag["projection_iterations"]),
                    }
                )
    return rows


def gram_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for n in args.n_values:
        y = np.linspace(-2.0, 2.0, int(n))
        x = args.mu_star + y
        grad = rattle._constraint_grad(x, args.mu_star, args.k)
        gram = float(np.dot(grad, grad))
        rows.append(
            {
                "n": int(n),
                "gram": gram,
                "sqrt_gram": float(np.sqrt(gram)),
                "grad_norm_per_sqrt_n": float(np.sqrt(gram) / np.sqrt(n)),
            }
        )
    return rows


def write_report(args: argparse.Namespace, gradient: pd.DataFrame, trajectory: pd.DataFrame, gram: pd.DataFrame) -> None:
    completed = trajectory[trajectory["stage"].eq("completed")].copy()
    if completed.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            completed.groupby(["eps", "n"], as_index=False)
            .agg(
                cases=("seed", "count"),
                failures=("ok", lambda s: int((~s.astype(bool)).sum())),
                mean_abs_delta_h=("abs_delta_h", "mean"),
                max_abs_delta_h=("abs_delta_h", "max"),
                max_constraint_residual=("proposal_constraint_residual", "max"),
                max_tangent_residual=("proposal_tangent_residual", "max"),
                max_reverse_position_error=("reverse_position_error", "max"),
                mean_forward_projection_iterations=("forward_projection_iterations", "mean"),
            )
        )
    summary.to_csv(args.out / "trajectory_summary.csv", index=False)

    lines = [
        "# Student RATTLE Algorithm Mechanics Audit",
        "",
        "This audit checks local RATTLE mechanics only; it is not a long-chain posterior validation or profiler.",
        "",
        "## Gradient Check",
        gradient.to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Gram Scaling Probe",
        gram.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Forward/Reverse Trajectory Summary",
        summary.to_markdown(index=False, floatfmt=".6g") if not summary.empty else "No completed trajectory rows.",
        "",
        "## Interpretation",
        "",
        "- Passing gradient/reverse/constraint checks supports the mechanics of the current paper-fixed-direction RATTLE map.",
        "- Energy error should increase with step size and often with n; if it does not, the tested step size is probably too conservative for this range.",
        "- A low wall-time scaling slope is not proof of dimension-free sampling; compare this audit with a profiler that decomposes Python overhead and vector kernel time.",
    ]
    (args.out / "algorithm_audit_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gradient = pd.DataFrame(gradient_rows(args))
    trajectory = pd.DataFrame(trajectory_rows(args))
    gram = pd.DataFrame(gram_rows(args))

    gradient.to_csv(args.out / "gradient_check.csv", index=False)
    trajectory.to_csv(args.out / "trajectory_checks.csv", index=False)
    gram.to_csv(args.out / "gram_scaling.csv", index=False)
    write_report(args, gradient, trajectory, gram)
    print(f"wrote audit to {args.out}", flush=True)


if __name__ == "__main__":
    main()
