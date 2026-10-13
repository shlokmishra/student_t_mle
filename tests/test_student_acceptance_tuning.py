from scripts.tune_student_acceptance import _aggregate, _select


def _row(method, n, mu_scale, latent_scale, mu_acceptance, latent_acceptance, healthy=True):
    return {
        "method": method,
        "n": n,
        "seed": 0,
        "proposal_std_mu": mu_scale,
        "proposal_std_z": latent_scale if method == "gibbs" else "",
        "rattle_step_size": latent_scale if method == "rattle" else "",
        "rattle_num_steps": 5 if method == "rattle" else "",
        "mu_acceptance_rate": mu_acceptance,
        "latent_acceptance_rate": latent_acceptance,
        "numerically_healthy": healthy,
        "projection_failure_count": 0 if healthy else 1,
        "reverse_check_failure_count": 0,
    }


def test_selection_uses_only_acceptance_distance():
    rows = [
        _row("gibbs", 10, 0.1, 0.01, 0.60, 0.60),
        _row("gibbs", 10, 0.2, 0.02, 0.45, 0.43),
        _row("rattle", 10, 0.1, 0.05, 0.43, 0.66),
        _row("rattle", 10, 0.2, 0.10, 0.44, 0.65, healthy=False),
    ]
    selected = _select(_aggregate(rows, target_mu=0.44, target_gibbs_z=0.44, target_rattle=0.65))
    by_method = {row["method"]: row for row in selected}
    assert by_method["gibbs"]["proposal_std_mu"] == 0.2
    assert by_method["gibbs"]["proposal_std_z"] == 0.02
    assert by_method["rattle"]["rattle_step_size"] == 0.05


def test_aggregation_averages_acceptance_across_seeds():
    rows = [
        _row("gibbs", 50, 0.2, 0.02, 0.4, 0.3),
        {**_row("gibbs", 50, 0.2, 0.02, 0.6, 0.5), "seed": 1},
    ]
    aggregate = _aggregate(rows, target_mu=0.5, target_gibbs_z=0.4, target_rattle=0.65)
    assert len(aggregate) == 1
    assert aggregate[0]["mean_mu_acceptance_rate"] == 0.5
    assert aggregate[0]["mean_latent_acceptance_rate"] == 0.4
    assert aggregate[0]["acceptance_score"] == 0.0
