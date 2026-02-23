# Gibbs for MLE (location models)

Simulate from the Bayesian posterior of theta (location) given the MLE for several location families. Two-step augmented Gibbs; verification via the KDE trick.

## Models

- **Student-t** (df k, scale 1)
- **Laplace** (scale b)
- **Logistic** (scale 1)

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md).

- `models/` — per-model: MLE, benchmark MLE samples, Gibbs sampler
- `kde_ref/` — reference posterior p(theta | hat_theta*) from KDE
- `analysis.py` — posterior_variance_from_kde, etc.
- `validation.py` — run comparison Gibbs vs KDE
- `run_compare.py` — entry point

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_compare.py --model student --k 2 --n 20
```

(Exact CLI to be defined.)
