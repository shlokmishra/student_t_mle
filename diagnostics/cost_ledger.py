"""Cost accounting helpers for Student-t Gibbs/RATTLE diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


COUNTER_DEFAULTS = {
    "wall_time_sec": 0.0,
    "iterations": 0,
    "n": 0,
    "k": np.nan,
    "mu_star": np.nan,
    "seed": 0,
    "method": "",
    "model": "",
    "supports_rattle": False,
    "rattle_status": "",
    "mle_convention": "",
    "target_description": "",
    "gibbs_backend": "",
    "num_iterations": 0,
    "burn_in": 0,
    "run_status": "",
    "source_file": "",
    "student_logpdf_evals": 0,
    "student_grad_evals": 0,
    "prior_logpdf_evals": 0,
    "prior_grad_evals": 0,
    "potential_evals": 0,
    "gradient_evals": 0,
    "constraint_evals": 0,
    "constraint_grad_evals": 0,
    "gram_evals": 0,
    "gram_grad_evals": 0,
    "projection_evals": 0,
    "projection_failures": 0,
    "max_constraint_abs": 0.0,
    "mean_constraint_abs": 0.0,
    "mu_mh_proposals": 0,
    "mu_mh_accepts": 0,
    "pair_updates_attempted": 0,
    "pair_updates_completed": 0,
    "pair_grid_evals": 0,
    "pair_inverse_branch_evals": 0,
    "pair_weight_evals": 0,
    "pair_rejections": 0,
    "block_z_accepts": 0,
    "block_z_acceptance_rate": np.nan,
    "sweep_count": 0,
    "hmc_proposals": 0,
    "hmc_accepts": 0,
    "leapfrog_steps": 0,
    "forward_newton_iters": 0,
    "reverse_newton_iters": 0,
    "momentum_projections": 0,
    "reverse_check_attempts": 0,
    "reverse_check_failures": 0,
    "energy_evals": 0,
    "metropolized_rejections": 0,
    "integration_failures": 0,
    "projection_mode": "",
    "position_projection_newton_iters": 0,
    "position_projection_failures": 0,
    "reverse_position_error": 0.0,
    "reverse_momentum_error": 0.0,
    "gram_correction_enabled": False,
}


OUTPUT_COLUMNS = [
    "method",
    "model",
    "n",
    "k",
    "mu_star",
    "seed",
    "iterations",
    "num_iterations",
    "burn_in",
    "run_status",
    "source_file",
    "wall_time_sec",
    "student_logpdf_evals",
    "student_grad_evals",
    "constraint_evals",
    "constraint_grad_evals",
    "gram_evals",
    "projection_evals",
    "projection_failures",
    "max_constraint_abs",
    "mean_constraint_abs",
    "mu_mh_proposals",
    "mu_mh_accepts",
    "pair_updates_attempted",
    "pair_updates_completed",
    "block_z_accepts",
    "block_z_acceptance_rate",
    "pair_grid_evals",
    "hmc_proposals",
    "hmc_accepts",
    "leapfrog_steps",
    "forward_newton_iters",
    "reverse_newton_iters",
    "reverse_check_failures",
    "projection_mode",
    "position_projection_newton_iters",
    "position_projection_failures",
    "reverse_position_error",
    "reverse_momentum_error",
    "gram_correction_enabled",
    "supports_rattle",
    "rattle_status",
    "mle_convention",
    "target_description",
    "gibbs_backend",
    "ess_mu",
    "ess_per_sec",
    "acceptance_rate",
]


@dataclass
class CostLedger:
    """Small mutable counter ledger with wall-clock timing support."""

    method: str
    n: int
    k: float
    mu_star: float
    seed: int
    model: str = ""
    iterations: int = 0
    counters: dict[str, Any] = field(default_factory=lambda: dict(COUNTER_DEFAULTS))
    _start_time: float | None = None
    _constraint_abs_sum: float = 0.0
    _constraint_abs_count: int = 0

    def __post_init__(self) -> None:
        self.counters.update(
            {
                "method": self.method,
                "model": self.model,
                "n": int(self.n),
                "k": float(self.k),
                "mu_star": float(self.mu_star),
                "seed": int(self.seed),
                "iterations": int(self.iterations),
            }
        )

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        if self._start_time is not None:
            self.counters["wall_time_sec"] = float(time.perf_counter() - self._start_time)
            self._start_time = None

    def inc(self, key: str, amount: int | float = 1) -> None:
        if key not in self.counters:
            self.counters[key] = 0
        self.counters[key] += amount

    def set(self, key: str, value: Any) -> None:
        self.counters[key] = value

    def observe_constraint(self, value: float) -> None:
        abs_value = abs(float(value))
        self._constraint_abs_sum += abs_value
        self._constraint_abs_count += 1
        self.counters["max_constraint_abs"] = max(float(self.counters.get("max_constraint_abs", 0.0)), abs_value)
        self.counters["mean_constraint_abs"] = self._constraint_abs_sum / max(self._constraint_abs_count, 1)

    def update_from_projection_diag(self, diag: dict[str, Any]) -> None:
        self.counters["projection_failures"] = int(
            diag.get("projection_failure_count", self.counters.get("projection_failures", 0))
        )
        self.counters["reverse_check_failures"] = int(
            diag.get("reverse_check_failure_count", self.counters.get("reverse_check_failures", 0))
        )
        self.counters["position_projection_failures"] = int(
            diag.get("position_projection_failures", self.counters.get("position_projection_failures", 0))
        )
        self.counters["max_constraint_abs"] = float(
            diag.get("max_manifold_residual", self.counters.get("max_constraint_abs", 0.0))
        )
        self.counters["mean_constraint_abs"] = float(
            diag.get("mean_manifold_residual", self.counters.get("mean_constraint_abs", 0.0))
        )
        self.counters["reverse_position_error"] = float(
            diag.get("max_reverse_position_error", self.counters.get("reverse_position_error", 0.0))
        )
        self.counters["reverse_momentum_error"] = float(
            diag.get("max_reverse_momentum_error", self.counters.get("reverse_momentum_error", 0.0))
        )

    def to_dict(self, **extra: Any) -> dict[str, Any]:
        row = dict(self.counters)
        row.update(extra)
        return row

    def output_row(self, **extra: Any) -> dict[str, Any]:
        row = self.to_dict(**extra)
        return {column: row.get(column, np.nan) for column in OUTPUT_COLUMNS}
