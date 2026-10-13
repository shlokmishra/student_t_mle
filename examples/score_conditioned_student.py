"""Small production-kernel smoke example; this does not validate selected-MLE sampling."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.loc.student import score_gibbs as loc_student


def main():
    params = dict(n=3, k=2.0, num_iterations_T=40, proposal_std_mu=0.5,
                  proposal_std_z=0.1, prior_mean=0.0, prior_std=10.0,
                  gibbs_backend="numba_full")
    key = np.array([0, 20260920], dtype=np.uint32)
    out = loc_student.run_gibbs(key, 0.0, params, verbose=False)
    x = out["x_chain"]
    residual = np.max(np.abs(np.sum(x / (params["k"] + x * x), axis=1)))
    assert x.shape == (41, 3)
    assert out["mu_chain"].shape == (41,)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(out["mu_chain"]))
    assert residual < 1e-9
    print(json.dumps({"target": "student_location_score", "release": 0.0,
                      "seed_key": key.tolist(), "parameters": params,
                      "mu_shape": list(out["mu_chain"].shape),
                      "x_shape": list(x.shape), "max_score_residual": float(residual),
                      "validation": "shape/finiteness/constraint smoke only"}, indent=2))


if __name__ == "__main__":
    main()
