import numpy as np
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_laplace_facet_rattle_chain_behavior.py"
SPEC = importlib.util.spec_from_file_location("validate_laplace_facet_rattle_chain_behavior", SCRIPT_PATH)
chain_behavior = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chain_behavior)


def test_chain_behavior_summary_reports_ess_and_split_drift():
    chain = np.linspace(-1.0, 1.0, 101)
    out = chain_behavior.chain_behavior_summary(chain, burnin=1, elapsed_seconds=2.0)
    assert out["num_samples"] == 100
    assert np.isfinite(out["ess_mu"])
    assert np.isfinite(out["ess_per_sec_mu"])
    assert out["ess_per_sec_mu"] == out["ess_mu"] / 2.0
    assert out["split_mean_drift"] > 0.0
    assert out["q025"] <= out["q50"] <= out["q975"]


def test_ess_stats_handles_constant_chain():
    out = chain_behavior.ess_stats(np.ones(20))
    assert np.isnan(out["ess_mu"])
    assert np.isnan(out["acf_lag1_mu"])
