import numpy as np
import pandas as pd
from argparse import Namespace

from reporting.diagnostics.diagnose_student_score_vs_mle import diagnostic_rows
from reporting.diagnostics.study_student_mle_transitions import mle_competitor_margin


def test_competitor_margin_is_zero_for_two_equal_global_modes():
    x = np.array([-10.0, -10.0, 10.0, 10.0])
    left = mle_competitor_margin(x, mu_star=-9.9498743710662, k=1.0)
    assert left["mu_star_is_global_mle"]
    assert left["num_local_maxima"] == 2
    assert abs(left["mle_margin"]) < 1e-8


def test_competitor_margin_rejects_stationary_local_minimum():
    x = np.array([-10.0, -10.0, 10.0, 10.0])
    center = mle_competitor_margin(x, mu_star=0.0, k=1.0)
    assert not center["mu_star_is_local_max"]
    assert not center["mu_star_is_global_mle"]
    assert center["mle_margin"] < 0.0


def test_competitor_margin_is_infinite_without_another_mode():
    x = np.array([-1.0, -0.5, 0.5, 1.0])
    result = mle_competitor_margin(x, mu_star=0.0, k=2.0)
    assert result["mu_star_is_global_mle"]
    assert result["num_local_maxima"] == 1
    assert np.isposinf(result["mle_margin"])


def test_endpoint_global_mle_diagnostic_does_not_shadow_score_function():
    frame = pd.DataFrame(
        [{
            "model": "student_t", "method": "gibbs", "k": 2.0, "n": 4,
            "mu_star": 0.0, "seed": 0, "iteration": 0, "mu": 0.0,
            "x_0": -1.0, "x_1": -0.5, "x_2": 0.5, "x_3": 1.0,
        }]
    )
    args = Namespace(
        model="student_t", k_values=[2.0], n_values=[4], score_tol=1e-8,
        mle_tol=1e-6, loglik_gain_tol=1e-8, dense_validation=False,
        dense_validation_stride=200, dense_points_per_interval=25,
    )
    out = diagnostic_rows(frame, args)
    assert out.loc[0, "mle_error"] == ""
    assert out.loc[0, "mu_star_is_global_mle"]
