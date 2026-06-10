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
  KDE should be treated as diagnostic smoothing only.

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

- Report: `results/final_production_v1_correctness_audit/sampler_correctness_report.md`
- Decision memo: `results/final_production_v1_correctness_audit/sampler_correctness_decision_memo.md`
- Verdicts: `results/final_production_v1_correctness_audit/final_sampler_verdict_table.csv`
- Suspicious cases: `results/final_production_v1_correctness_audit/suspicious_sampler_cases.csv`
- Figures: `results/final_production_v1_correctness_audit/figures/`

Slide candidate: "Sampler correctness is judged against raw weighted-MC, not KDE"

Main claim:

- Gibbs and RATTLE are audited numerically against raw weighted-MC posterior summaries.
- This update uses the final production runset: 100k iterations, 20k burn-in, 3 seeds, thinned transition/geometry diagnostics.
- Laplace scalar median comparisons use odd n=11,21,51, and Laplace RATTLE is not applicable.
- Student-t k=1,n=10 remains unresolved; k=1 more broadly needs caution.

Key numbers:

- Production completeness: 81/81 case directories completed with 100k iterations,
  20k burn-in, and 50-step diagnostic thinning; 0 missing required output rows
  and 0 failed case rows.
- Verdict counts: `{'pass_with_warning': 14, 'pass': 14, 'unresolved': 2}`.
- Final dashboard verdicts: 14 clean, 11 caveat-only, 2 unresolved, and 3
  not-applicable Laplace RATTLE rows.
- Posterior agreement status counts: `{'pass': 14, 'warning': 13, 'not_applicable': 3}`.
- High-severity suspicious sampler cases: `5`.
- Clean Student-t examples: k=2,n=50 and k=3,n=20,50 for both Gibbs and
  RATTLE.
- Logistic is clean for Gibbs and RATTLE at n=10,20,50.
- Laplace Gibbs is clean at n=21,51; n=11 is caveat-only from a tail/quantile
  posterior warning; Laplace RATTLE is not applicable.
- Student-t k=1,n=10 remains unresolved for both methods. Student-t k=1 at
  n=20,50 is caveat-only, not a headline clean result.
- Student-t k=2,n=10 and k=2,n=20 are caveat-only after production-length runs;
  this is no longer a smoke-contamination artifact.

Diagnostic headline:

- Gibbs constraint residuals are numerically small across production cases:
  max cached residual is about 1.2e-9, below the 1e-6 tolerance.
- Student Gibbs branch diagnostics are available and use all four branch-pair
  categories in the audited Student cases.
- RATTLE geometry is clean wherever applicable: projection failure and reverse
  failure rates are zero; position/tangent residuals are about 1e-10.
- RATTLE energy is controlled: max `|delta_H|` is about 0.375 in the hardest
  Student-t k=1,n=10 case; for Student-t k=2,3 it is at most about 0.033, and
  for Logistic at most about 0.0013.

Plots worth showing:

- `results/final_production_v1_correctness_audit/figures/posterior_agreement_heatmap.png`
- `results/final_production_v1_correctness_audit/figures/ess_per_sec_heatmap.png`
- `results/final_production_v1_correctness_audit/figures/rel_sd_error_heatmap.png`
- `results/final_production_v1_correctness_audit/figures/rattle_constraint_residual_plot.png`
- `results/final_production_v1_correctness_audit/figures/rattle_delta_H_histogram.png`
- `results/final_production_v1_correctness_audit/figures/student_branch_usage_plot.png`

Collaborator caveat:

- RATTLE geometry/reversibility diagnostics are clean where applicable: projection/reverse failures are zero, constraints/tangency are near numerical zero, and delta_H is controlled.
- Gibbs constraints are clean, and Student branch usage is cached. The available pair-delta column compares thinned snapshots, so it is not a direct before/after pair-update invariant proof.
- For correctness claims, use `safe_to_present == yes` as clean examples,
  `caveat_only` for sensitivity/caveat slides, and keep `safe_to_present == no`
  out of headline Gibbs/RATTLE comparisons.

## Efficiency Audit

Source artifacts:

- Report: `results/final_production_v1_efficiency_audit_cost_first/efficiency_report.md`
- Aggregate table: `results/final_production_v1_efficiency_audit_cost_first/efficiency_summary.csv`
- Winner table: `results/final_production_v1_efficiency_audit_cost_first/method_winners.csv`
- Seed-level cost decomposition: `results/final_production_v1_efficiency_audit_cost_first/cost_decomposition.csv`

Slide candidate: "Efficiency means reliable posterior information per second"

Main claim:

- Efficiency is analyzed from final production chains as cost per ESS for `mu`,
  with split/posterior warnings attached as caveats.
- Correctness caveats do not hide cost behavior; they determine whether a row is
  headline-clean, caveat cost-only, or diagnostic-only.
- RATTLE has higher `mu` ESS/sec in every comparable Gibbs/RATTLE production
  regime; Gibbs remains the only valid Laplace method.

Key numbers:

- Seed-level cost rows: 81; aggregate rows: 27 model/k/n/method summaries.
- Winner rows: 15 model/k/n regimes.
- Main-claim winner counts: `{'rattle': 7, 'gibbs_only': 3}`.

Plots/tables worth showing:

- `results/final_production_v1_efficiency_audit_cost_first/method_winners.csv` for the "who wins and when" table.
- `results/final_production_v1_efficiency_audit_cost_first/efficiency_summary.csv` for ESS/sec, sec/iteration, wall-time per
  ESS, split drift, and posterior-warning context.

Reasoning to mention:

- Use ESS/sec and wall-time per ESS, not raw runtime alone.
- Cost-only caveat rows are useful for engineering expectations after tuning,
  but should not become headline scientific correctness examples.
- Laplace is Gibbs-only because RATTLE is not applicable for the nonsmooth median
  target.

## Geometry Audit

Source artifacts:

- Report: `results/final_production_v1_geometry_audit/geometry_report.md`
- Geometry summary: `results/final_production_v1_geometry_audit/geometry_summary.csv`
- Geometry-conditioned posterior: `results/final_production_v1_geometry_audit/geometry_conditioned_posterior.csv`
- Win/loss table: `results/final_production_v1_geometry_audit/geometry_win_loss_table.csv`
- Figures: `results/final_production_v1_geometry_audit/figures/`
- Focused Cauchy runset config: `configs/student_k1_n50_geometry_cases.yaml`
- Focused Cauchy outputs: `results/student_k1_n50_geometry_runs/`
- Focused Cauchy audit: `results/student_k1_n50_geometry_audit/`

Slide candidate: "Geometry explains when RATTLE wins and why k=1 is hard"

Main claim:

- RATTLE wins efficiency in the clean comparable regimes because its geometric
  moves traverse smooth/manageable latent manifolds cheaply and with clean
  projection/reversibility diagnostics.
- Gibbs is correct and branch-aware, but its local pair moves become less
  competitive as dimension/tail geometry grows.
- Student-t k=1 is not just a tuning problem: the latent geometry is dominated
  by extreme-tail states, especially for Gibbs at larger n.

Key numbers:

- Geometry rows: 129,681 thinned latent states; 27 model/k/n/method summaries.
- Student-t k=1 is extreme-tail dominated: dominant extreme-tail fraction is
  about 0.86/0.98/1.00 for Gibbs at n=10/20/50 and 0.84/0.95/1.00 for RATTLE.
- Student-t k=1 Gibbs has enormous max-|x| behavior: mean max-|x| is about 65,
  1,975, and 2,011 for n=10,20,50, with maxima up to about 8.4e6.
- Student-t k=2 and k=3 are much more manageable: mean max-|x| is mostly about
  3-13, with RATTLE generally lower than Gibbs.
- Gibbs Student branch diagnostics use all four branch-pair categories. Mean
  branch switching is about 0.42/0.50/0.50 for k=1 at n=10/20/50, about
  0.39-0.41 for k=2, and about 0.28 for k=3.
- Gibbs latent-class switching flags k=1 as sticky at larger n: class switching
  is about 0.20, 0.04, and 0.0004 for k=1 at n=10,20,50.
- RATTLE projection/reverse failures remain zero in all applicable geometry
  rows; delta-H/tail diagnostics are available and do not indicate geometry
  failures.

Plots worth showing:

- `results/final_production_v1_geometry_audit/figures/student_tail_geometry_histogram.png`
- `results/final_production_v1_geometry_audit/figures/student_branch_occupancy.png`
- `results/final_production_v1_geometry_audit/figures/latent_geometry_class_transition_heatmap.png`
- `results/final_production_v1_geometry_audit/figures/rattle_delta_H_vs_max_abs_y.png`
- `results/final_production_v1_geometry_audit/figures/geometry_win_loss_summary_heatmap.png`

Reasoning to mention:

- Logistic is the smooth benchmark: RATTLE is clean and fast, Gibbs is correct
  but slower.
- Student k=2,3 show manageable tail geometry; RATTLE wins the cost comparison
  while maintaining clean geometry checks in the headline-clean cases.
- Student k=1 should be presented as a difficult heavy-tail geometry case, not a
  headline Gibbs/RATTLE comparison.
- Laplace is a nonsmooth median/order-statistic geometry case, so RATTLE remains
  not applicable and Gibbs is the presentation baseline.

Focused Student k=1,n=50 Gibbs follow-up:

- Run 15 long chains on Grace: 5 seeds x central/tail-heavy/random
  initializations, 500k iterations, 100k burn-in, diagnostic thinning 100.
- Save full thinned latent snapshots (`x_0` through `x_49`) so the audit can
  measure score-coordinate collapse `z=y/(1+y^2)`, branch occupancy, extreme
  tail thresholds, and geometry-conditioned `mu` summaries.
- Generate jobs with
  `python scripts/make_grace_targeted_validation_jobs.py --case-config configs/student_k1_n50_geometry_cases.yaml --out-tsv hpc/grace/student_k1_n50_geometry_cases.tsv --out-dir results/student_k1_n50_geometry_runs --submit-script hpc/grace/submit_student_k1_n50_geometry.sh --save-full-latent-diagnostics`.
- Refresh with `python reporting/diagnostics/analyze_geometry.py --runsets student_k1_n50_geometry --out-dir results/student_k1_n50_geometry_audit`.
- Success means a defensible heavy-tail geometry explanation, not necessarily
  removing all Student k=1 caveat labels.

## MLE Release Information / Privacy Payoff

Source artifacts:

- Runset: `results/release_information_runs/`
- Report: `results/final_production_v1_release_information_audit/release_information_report.md`
- Decision memo: `results/final_production_v1_release_information_audit/release_information_decision_memo.md`
- Information-loss table: `results/final_production_v1_release_information_audit/information_loss_summary.csv`
- Privacy-leakage table: `results/final_production_v1_release_information_audit/privacy_leakage_summary.csv`
- Observed-outlier table: `results/final_production_v1_release_information_audit/observed_outlier_summary.csv`
- Figures: `results/final_production_v1_release_information_audit/figures/`

Slide candidate: "What does releasing only the MLE preserve, distort, and leak?"

Main claim:

- Normal-known-variance is the zero-information-loss benchmark: the mean is
  sufficient, so full-data Bayes and MLE-only Bayes agree exactly.
- Logistic and Student k=2,3 are close but not exactly sufficient: MLE-only
  posteriors usually have slightly wider uncertainty and small Wasserstein
  shifts, shrinking with n.
- Heavy-tail/small-n cases, especially Student k=1,n=10, show the real risk:
  the same MLE can hide very different tail configurations, so MLE-only Bayes
  can lose important dataset-specific tail information.

Key numbers:

- Paired information-loss runset: 540 simulated datasets = 18 regimes x 30
  datasets.
- MLE-only posterior used raw weighted Monte Carlo from 30,000 centered-MLE
  simulations per regime.
- Normal benchmark: median SD ratio = 1.000 and Wasserstein/quantile differences
  are numerical zero for n=10,20,50.
- Logistic median SD ratios are about 1.041, 1.008, 1.010 for n=10,20,50;
  median Wasserstein drops from about 0.030 to 0.009.
- Student k=2 median SD ratios are about 1.058, 1.009, 1.006 for n=10,20,50.
- Student k=3 median SD ratios are about 1.038, 0.989, 1.004 for n=10,20,50.
- Student k=1,n=10 is the standout information-loss case: median SD ratio
  about 1.54, median Wasserstein about 0.112, and max quantile-distance about
  17.3 across the 30 datasets.
- Actual simulated Student k=1 max-|x - hat_mu| is large: median 8.39, 17.73,
  50.15 for n=10,20,50; 95th percentiles about 213, 239, 414.
- Production latent privacy chains at `hat_mu=0` show substantial compatible
  extreme-outlier uncertainty. For Student k=1,n=10, posterior P(M>10 | hat_mu)
  is about 0.567 for both Gibbs and RATTLE versus prior about 0.481.

Plots worth showing:

- `results/final_production_v1_release_information_audit/figures/sd_ratio_heatmap.png`
- `results/final_production_v1_release_information_audit/figures/wasserstein_heatmap.png`
- `results/final_production_v1_release_information_audit/figures/interval_width_ratio_heatmap.png`
- `results/final_production_v1_release_information_audit/figures/privacy_leakage_probability_shift.png`
- `results/final_production_v1_release_information_audit/figures/posterior_extreme_probability_by_threshold.png`
- `results/final_production_v1_release_information_audit/figures/representative_information_loss_cases.png`

Reasoning to mention:

- Correctness/efficiency/geometry were prerequisites; this section is the
  statistical/privacy payoff.
- Information loss is measured by paired full-data vs MLE-only posterior
  summaries, not visual overlay alone.
- Privacy leakage is framed as prior-to-posterior belief shift about latent
  extremes compatible with the released MLE, not individual re-identification.
- Student k=1,n=10 remains a diagnostic/caveat example because correctness and
  geometry already mark it as difficult.
