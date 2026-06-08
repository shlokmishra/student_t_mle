# Presentation Notes

Living notes for the final analysis presentation. Keep entries short enough to
transfer into slides, and link to the exact supporting artifact when possible.

## How to Use This File

- Add one section per analysis step.
- For each step, record the slide-level claim, 2-5 key numbers, plots worth
  showing, and the collaborator-facing interpretation.
- Avoid pasting full tables; link to the CSV/report instead.

## KDE Correctness Audit

Source artifacts:

- Report: `results/kde_correctness_audit/kde_correctness_report.md`
- Recommendations: `results/kde_correctness_audit/backend_recommendations.csv`
- Suspicious cases: `results/kde_correctness_audit/suspicious_kde_cases.csv`
- Figures: `results/kde_correctness_audit/figures/`

Slide candidate: "KDE is diagnostic, raw weighted-MC is the benchmark"

Main claim:

- KDE backends are stable enough for smoothed density visualization for
  Logistic, Laplace odd-n, and Student-t k=2,3.
- Student-t k=1 is not stable enough for a single KDE backend at any audited n.
- Scientific posterior summaries should use raw weighted-MC as the benchmark;
  KDE should not be treated as ground truth.

Key numbers:

- 45 cached posterior cases audited across Scott, SJ_transform, and t_abram.
- 15 model/k/n recommendation groups.
- 12 of 15 groups recommend `scott`.
- 3 of 15 groups recommend `no_single_kde_recommended`: Student-t k=1 at
  n=10,20,50.
- In stable non-k=1 cases, Scott-vs-SJ relative SD differences are small:
  about 0.0030 to 0.0056.
- Student-t k=1 Scott-vs-SJ relative SD differences are large:
  n=10 mean about 0.34, n=20 mean about 0.70, n=50 mean about 1.02.
- Student-t k=1 Scott tail-mass errors are material:
  n=10 about 0.056 to 0.143, n=20 about 0.257 to 0.354, n=50 about 0.599 to
  0.718.
- `t_abram` is uncapped/full-B in this cache (`available_uncapped_tail_stress_test`);
  still use it only as a tail stress-test backend, not as the primary default.

Backend recommendation summary:

- Laplace n=11,21,51: default to `scott`; `SJ_transform` is secondary.
- Logistic n=10,20,50: default to `scott`; `SJ_transform` is secondary.
- Student-t k=2 and k=3 at n=10,20,50: default to `scott`;
  `SJ_transform` is secondary.
- Student-t k=1 at n=10,20,50: no single KDE backend recommended.

Plots worth showing:

- `results/kde_correctness_audit/figures/backend_summary_error_heatmap.png`
  to summarize where backend error concentrates.
- `results/kde_correctness_audit/figures/tail_probability_error_heatmap.png`
  to show Student-t k=1 tail instability.
- `results/kde_correctness_audit/figures/density_overlay_student_t_k1_n10.png`
  as the cautionary example.
- `results/kde_correctness_audit/figures/cdf_overlay_student_t_k1_n10.png`
  to explain why tail/CDF diagnostics matter beyond visual density overlays.
- One representative stable overlay, for example
  `results/kde_correctness_audit/figures/density_overlay_logistic_kna_n20.png`
  or `results/kde_correctness_audit/figures/density_overlay_student_t_k2_n20.png`.

Reasoning to mention:

- Scott is selected where applicable because it has a smaller combined score
  than SJ_transform and no serious backend/seed/tail flags.
- Student-t k=1 is a KDE diagnostic failure mode, not necessarily a Gibbs/RATTLE
  failure mode. Final Gibbs/RATTLE comparisons should be anchored to raw
  weighted-MC posterior summaries.
- Laplace KDE comparisons use only odd n=11,21,51. The even-n Laplace interval
  target is a separate analysis and should not be mixed into this KDE audit.

Dashboard default rule:

- Use raw weighted-MC as the posterior-summary benchmark.
- Default smooth KDE visualization to `scott` for Logistic, Laplace odd-n, and
  Student-t k=2,3.
- For Student-t k=1 at all audited n, show a warning and avoid a primary KDE
  default; offer KDE only as diagnostic/sensitivity visualization.
- Keep `SJ_transform` as secondary sensitivity and `t_abram` as optional tail
  stress test.

Collaborator caveat:

- Do not present KDE as the reference posterior. It is a smoothed visualization
  layer. If a conclusion changes depending on KDE backend, report the raw
  weighted-MC result and flag the KDE sensitivity.

## Gibbs/RATTLE Correctness Audit

Source artifacts:

- Report: `results/sampler_correctness_audit/sampler_correctness_report.md`
- Verdicts: `results/sampler_correctness_audit/sampler_correctness_summary.csv`
- Suspicious cases: `results/sampler_correctness_audit/suspicious_sampler_cases.csv`
- Figures: `results/sampler_correctness_audit/figures/`

Slide candidate: "Sampler correctness is judged against raw weighted-MC, not KDE"

Main claim:

- Gibbs and RATTLE are audited numerically against raw weighted-MC posterior summaries.
- Laplace scalar median comparisons use odd n=11,21,51, and Laplace RATTLE is not applicable.
- Student-t k=1,n=10 remains unresolved; k=1 more broadly needs caution.

Key numbers:

- Verdict counts: `{'unresolved': 26, 'pass_with_warning': 4}`.
- High-severity suspicious sampler cases: `1`.

Plots worth showing:

- `results/sampler_correctness_audit/figures/posterior_agreement_heatmap.png`
- `results/sampler_correctness_audit/figures/ess_per_sec_heatmap.png`
- `results/sampler_correctness_audit/figures/rel_sd_error_heatmap.png`
- `results/sampler_correctness_audit/figures/rattle_constraint_residual_plot.png`
- `results/sampler_correctness_audit/figures/student_score_vs_selected_mle_mismatch_plot.png`

Collaborator caveat:

- Missing low-level pair-branch, delta_H, and momentum diagnostics mean the audit is strong on posterior/ledger/chain behavior but not a full transition-level proof.
