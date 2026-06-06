"""Registry of location models and comparison applicability metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    parameter_name: str
    supports_kde_reference: bool
    supports_gibbs: bool
    supports_rattle: bool
    supports_cost_audit: bool
    mle_type: str
    mle_convention: str
    constraint_type: str
    default_parameters: dict[str, Any]
    target_description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "student_t": ModelSpec(
        model_name="student_t",
        parameter_name="mu",
        supports_kde_reference=True,
        supports_gibbs=True,
        supports_rattle=True,
        supports_cost_audit=True,
        mle_type="smooth_score_root",
        mle_convention="root of sum_i (x_i - mu_hat)/(k + (x_i - mu_hat)^2) = 0; k required",
        constraint_type="smooth_score",
        default_parameters={"k": 2.0, "n": 20, "prior_mean": 0.0, "prior_std": 10.0},
        target_description="Condition on Student-t location score equation at observed mu_star.",
    ),
    "logistic": ModelSpec(
        model_name="logistic",
        parameter_name="mu",
        supports_kde_reference=True,
        supports_gibbs=True,
        supports_rattle=True,
        supports_cost_audit=True,
        mle_type="smooth_score_root",
        mle_convention="root of sum_i tanh((x_i - mu_hat)/2) = 0",
        constraint_type="smooth_score",
        default_parameters={"n": 20, "prior_mean": 0.0, "prior_std": 10.0},
        target_description="Condition on logistic location score equation at observed mu_star.",
    ),
    "laplace": ModelSpec(
        model_name="laplace",
        parameter_name="mu",
        supports_kde_reference=True,
        supports_gibbs=True,
        supports_rattle=False,
        supports_cost_audit=True,
        mle_type="median",
        mle_convention="unique sample median for odd n; numpy median averages middle two for even n",
        constraint_type="nonsmooth_median",
        default_parameters={"n": 21, "n_values": [11, 21, 51], "b": 1.0, "prior_mean": 0.0, "prior_std": 10.0},
        target_description="deterministic_median_equals_mu_star",
    ),
}

LAPLACE_NP_MEDIAN_TARGET = {
    "target_id": "laplace_np_median_target",
    "target_description": "deterministic_median_equals_mu_star",
    "mle_convention": "unique sample median for odd n; numpy median convention otherwise",
}

LAPLACE_MEDIAN_INTERVAL_TARGET = {
    "target_id": "laplace_median_interval_target",
    "target_description": "median_interval_contains_mu_star",
    "mle_convention": "lower and upper middle order statistics bracket mu_star for even n",
}


def get_model_spec(model: str) -> ModelSpec:
    key = str(model).lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model}")
    return MODEL_REGISTRY[key]


def laplace_target_for_n(n: int) -> dict[str, str]:
    return LAPLACE_NP_MEDIAN_TARGET if int(n) % 2 == 1 else LAPLACE_MEDIAN_INTERVAL_TARGET


def model_validity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, spec in MODEL_REGISTRY.items():
        methods = ["raw_weighted_mc", "kde", "gibbs", "rattle"]
        for method in methods:
            implementation_exists = True
            target_defined = True
            smooth_constraint = spec.constraint_type == "smooth_score"
            rattle_applicable = bool(spec.supports_rattle)
            target_matches_reference = True
            warnings = ""
            if method == "rattle" and not spec.supports_rattle:
                implementation_exists = False
                target_defined = False
                target_matches_reference = False
                warnings = "Exact RATTLE not applicable: Laplace median constraint is nonsmooth/order-based."
            if model == "laplace" and method == "gibbs":
                warnings = "For even n, Laplace Gibbs uses median_interval_contains_mu_star; odd-n default uses deterministic_median_equals_mu_star."
            rows.append(
                {
                    "model": model,
                    "k": "1,2,3" if model == "student_t" else "",
                    "method": method,
                    "implementation_exists": implementation_exists,
                    "target_description": spec.target_description,
                    "mle_convention": spec.mle_convention,
                    "target_defined": target_defined,
                    "target_matches_reference": target_matches_reference,
                    "smooth_constraint": smooth_constraint,
                    "rattle_applicable": rattle_applicable if method == "rattle" else "",
                    "tests_passed": "not_run",
                    "warnings": warnings,
                    "warning": warnings,
                }
            )
    return rows
