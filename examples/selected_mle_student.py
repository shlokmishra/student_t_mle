"""Tiny selected-global-MLE block-Gibbs smoke example, not a mixing study."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.loc.student import selected_block


def main():
    params = dict(n=3, k=2.0, num_iterations_T=40, block_length=4,
                  initial_x=np.array([-0.4, 0.4, 0.0]), initial_mu=0.0,
                  proposal_std_mu=0.45, proposal_std_z=0.1,
                  prior_mean=0.0, prior_std=2.0, n3_endpoint_fast_path=True)
    out = selected_block.run_gibbs(20261012, 0.0, params, verbose=False)
    assert out["mu_chain"].shape == (41,)
    assert out["x_chain"].shape == (41, 3)
    assert np.isfinite(out["mu_chain"]).all() and np.isfinite(out["x_chain"]).all()
    assert out["max_score_residual"] < 1e-9
    assert out["max_selected_mle_residual"] <= 1e-9
    print(json.dumps(dict(target=out["target"], seed=20261012,
        max_score_residual=out["max_score_residual"],
        max_selected_mle_residual=out["max_selected_mle_residual"],
        counts=out["counts"], validation="shape/finiteness/support smoke only"), indent=2))


if __name__ == "__main__":
    main()
