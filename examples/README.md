# Runnable smoke examples

Run from the repository root with the project Python:

```bash
.venv/bin/python examples/selected_mle_student.py
.venv/bin/python examples/score_conditioned_student.py
```

The first demonstrates the current selected-global-MLE block sampler with the
optional guarded n=3 check enabled explicitly. The second demonstrates the
score-conditioned comparison kernel. Both check short-chain shapes, finiteness
and constraints. Neither establishes posterior convergence or mixing.
