import numpy as np

from models import loc_laplace, loc_logistic
from models.model_registry import MODEL_REGISTRY, model_validity_rows


def test_model_registry_applicability_flags():
    assert MODEL_REGISTRY["student_t"].supports_rattle is True
    assert MODEL_REGISTRY["logistic"].supports_rattle is True
    assert MODEL_REGISTRY["laplace"].supports_rattle is False
    assert MODEL_REGISTRY["laplace"].constraint_type == "nonsmooth_median"


def test_logistic_transform_inverse_and_jacobian():
    y = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    z = np.tanh(y / 2.0)
    recovered = 2.0 * np.arctanh(z)
    derivative = 0.5 * (1.0 - np.tanh(y / 2.0) ** 2)
    assert np.allclose(recovered, y)
    assert np.all(derivative > 0.0)
    assert np.all(np.abs(z) < 1.0)


def test_laplace_get_mle_uses_numpy_even_median_convention():
    x = np.array([-2.0, -1.0, 3.0, 10.0])
    assert loc_laplace.get_mle(x, {"n": 4, "b": 1.0}) == np.median(x)
    assert loc_laplace.get_mle(x, {"n": 4, "b": 1.0}) == 1.0


def test_laplace_validity_matrix_defaults_to_odd_unique_median_target():
    rows = model_validity_rows()
    laplace_gibbs = [row for row in rows if row["model"] == "laplace" and row["method"] == "gibbs"][0]
    laplace_rattle = [row for row in rows if row["model"] == "laplace" and row["method"] == "rattle"][0]
    assert laplace_gibbs["target_description"] == "deterministic_median_equals_mu_star"
    assert laplace_gibbs["target_matches_reference"] is True
    assert "odd-n default" in laplace_gibbs["warnings"]
    assert laplace_rattle["rattle_applicable"] is False
    assert "not applicable" in laplace_rattle["warnings"]


def test_logistic_score_zero_at_mle_smoke():
    x = np.array([-2.0, -0.5, 0.25, 1.5, 3.0])
    mu_hat = loc_logistic.get_mle(x, {"n": x.size})
    score = np.sum(np.tanh((x - mu_hat) / 2.0))
    assert abs(score) < 1e-8
