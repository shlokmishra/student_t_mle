# Repository consolidation: Phase 1 audit and cleanup blueprint

Date: 2026-10-10. Audience: project collaborators, including Christian Robert and Antoine. **Proposal only: no cleanup has been executed.** Known-scale location inference is the immediate priority; scale and location-scale work remain preserved research.

## 1. Current repository architecture

The repository contains a sound and increasingly well-tested selected-MLE research path alongside an older score-conditioning experiment/reporting path. These paths are not interchangeable. The largest immediate risk is presenting their outputs under the same statistical label, rather than the physical location of the files.

### Inspection scope and working-tree protection

This audit read the README, checkpoint package, roadmap, evidence ledger, all twelve top-level research reports, relevant mathematical notes, model implementations, callers, reference helpers, configurations and evidence manifests. Jevgrep was used for behavioral discovery; exact searches and Python AST inspection established imports and call sites. No applicable `AGENTS.md` or root `MEMORY.md` was found in the inspected checkout/ancestor paths. Personal project memory helped locate the September checkpoint; current files were used to assess its relevance.

Starting Git HEAD: `b4f66744158d6517e1cb5bfb628f770eb1ead743`. Before adding this document there were **165 tracked files, 32 modified tracked files and 722 untracked files**. The newer selected-MLE modules, reports, tests and much of their evidence are untracked. Their scientific importance does not depend on Git status. All 887 tracked/untracked regular-file contents were hashed to an external scratch snapshot for an end-of-audit preservation comparison. No new archive was made.

The following counts exclude bytecode/cache directories, environments and selected third-party directory names; sizes are logical bytes, not allocated disk usage. They describe inspected roots, not a complete disk backup or a symlink-aware recovery manifest.

| Root | Files | Logical size | Current role |
|---|---:|---:|---|
| `models/` | 22 | 274,314 B | Models, selectors, shared math, score and selected-MLE samplers mixed together |
| `kde_ref/` | 5 | 38,593 B | Raw parameter references, KDE, cache/summary adapter |
| `diagnostics/` | 3 | 17,086 B | Run registry and cost accounting |
| `reporting/` | 61 | 897,574 B | Audits, proof utilities, historical presentations; some code imported by tests |
| `scripts/` | 56 | 545,638 B | Research drivers, validation, timing, job creation and preservation |
| `tests/` | 29 | 116,055 B | 28 test modules plus package file |
| `configs/`, `hpc/`, `examples/` | 31 | 38,387 B | Eight configurations, 22 cluster files, one score-conditioned example |
| `dashboard/` | 15 | 201,331 B | Historical Streamlit/reporting workflow |
| `docs/` | 772 | 149,170,025 B | Derivations, reports, compact evidence, decks, figures and checkpoint metadata |
| `results/` | 16,010 | 7,909,430,414 B | Ignored experimental outputs, including original chains |
| `tmp/` | 2,800 | 751,120,423 B | Ignored **scientific banks/chains**, failed-run records, presentation intermediates and dependency trees |
| `_archive_repo_cleanup_20260920_checkpoint/` | 11 | 9,051,650,359 B | Existing ignored recovery archive, Git bundle, patches and manifest |

Most storage is already concentrated in identifiable studies: `results/student_k1_n50_geometry_runs/` is approximately 3,612 MiB, `results/final_production_v1/` 1,709 MiB, and `results/release_information_runs_100_per_regime/` 1,238 MiB. `tmp/presentation_long_chains/` is 108 MiB and contains scientific evidence. These are preservation candidates, not deletion candidates.

The checkout is not installable: no `pyproject.toml`, `setup.py` or `setup.cfg` exists. Modules depend on repository-root imports. `models/__init__.py` eagerly imports legacy location and joint Student modules, bringing JAX initialization into otherwise small imports. New selected-MLE modules remain directly importable despite not being listed in `__all__`; this is not a broken import.

### Lightweight checks performed here

- AST inspection of 180 application/test Python files found no parse errors; explicit imports rooted in local packages resolved to existing files/directories. There are 193 statically named test functions, **not** 193 collected pytest cases: parametrization changes that count.
- Relative Markdown links in all twelve canonical research reports resolved. The all-model reference CSV referenced by every runset in `configs/analysis_run_registry.yaml` is absent.
- A selected set of evidence-manifest entries was checked for presence and recorded size: 126 unique entries across Gibbs, independence/geometry and n=5 manifests had no missing paths or size discrepancies. This is not a complete manifest hash verification.
- The fresh n=3 bank and both fresh shifted n=10 banks were independently hashed and matched recorded values; details appear in section 6.
- Installed distribution metadata was inspected without importing numerical kernels. No pytest, JIT compilation, sampler, figure regeneration or experiment was run. Historical test passes below are attributed to their reports, not rerun evidence. Runtime import success, fresh installation and full-suite health remain future gates.

## 2. Research component inventory

### Numerical implementations and actual users

| Component | Files and actual call paths | Target/status to retain |
|---|---|---|
| Student score Gibbs | `models/loc_student.py`; called by `scripts/run_cost_audit.py`, `run_targeted_validation.py`, `run_final_production_case.py`, profiling/tuning/geometry scripts and the example | Score-conditioned legacy comparator. `get_mle` using a global selector does **not** change its chain target. Numerical cutoffs remain part of this implementation. |
| Student score RATTLE | `models/loc_student_rattle.py`; same cost/production dispatch, reporting geometry and presentation studies | Score-coarea guiding target with Gram term; historical numerical reverse checks. Preserve as a comparator and shared integration source. |
| Transparent selected-MLE pair Gibbs | `models/loc_student_selected_mle.py`; `scripts/validate_student_selected_mle_gibbs.py`, selected-RATTLE validation, block kernel and focused tests | Joint pair proposal includes branch Jacobians, score-proposal Hastings ratio, selection indicator and curvature. Readable correctness baseline. |
| Compiled selected-MLE block Gibbs | `models/loc_student_selected_mle_block.py`; cheap-Gibbs, n=3 geometry/fast-path and independence validation drivers | Fixed-parameter random-scan regular score kernel raised to a fixed block length, followed by one endpoint correction. Preferred constrained implementation on tested domains. It is **not** an arbitrary legacy sweep with a new accept step. |
| Guarded n=3 endpoint | `models/student_n3_endpoint.py`, used by block sampler | Optional interval/root-location certificate; fallback to existing validation. `n3_endpoint_fast_path=False` remains the default. |
| Selected-MLE RATTLE | `models/loc_student_selected_mle_rattle.py`; its validation driver/tests; also supplies `geometry` and `selected_state` to block Gibbs | Correct regular-region endpoint Hamiltonian correction with stronger phase reversal checks. n=10 latent mixing remains inadequate. Shared target helpers currently live inside a sampler module. |
| Local affine MH | `models/loc_affine.py`; affine experiment, reference validation and shifted follow-up | Local Gaussian contrast-space MH plus parameter MH. Selected-MLE argument is justified under equivariance; shifted latent mixing fails. `run_gibbs` is an established but ambiguous API name. |
| Independence MH | `models/loc_student_selected_mle_independence.py`; independence validation/tests | Independent standardized datasets, prior-only MH ratio, cached current estimator. Both tested n=10 cases pass bounded comparisons; shifted tail efficiency trails block Gibbs. |
| Location selector | `models/student_location_mle.py`; Student/affine wrappers, selected samplers, validation/proof drivers | Hardened global selection, smaller location for **proved exact ties**, selective rational fallback and explicit errors. Not universal certification. |
| Shared parameter/math utilities | `models/student_mu_mh.py`, `numba_math.py`; Student variants, affine, Student joint code and Logistic compiled code | Preserve proposal law, RNG conventions and numerical guards; not newly standardized during cleanup. |
| Logistic location | `loc_logistic.py`, `logistic_gibbs_numba.py`, `loc_logistic_rattle.py`; registry/cost runners and reference audits | Score Gibbs/RATTLE comparators. Unique score root equals unique MLE **as a set**, but curvature varies, so conditional measures generally differ. |
| Laplace location | `loc_laplace.py`, `laplace_gibbs_numba.py`; cost/reference drivers | Odd-n fixed-label median chart; appropriate symmetric summaries do not imply exchangeability of labeled latent draws. Even-n interval-conditioning target differs from NumPy midpoint median. |
| Laplace facet RATTLE | `loc_laplace_rattle.py`; both `validate_laplace_facet_rattle*.py` drivers and focused tests | Explicit experimental nonsmooth fixed-facet method. Registry exclusion from ordinary smooth RATTLE is deliberate, not evidence that this file is unused. |
| Student location-scale | `loc_scale_student.py`, `loc_scale_student_rattle.py`; eager package exports and `test_loc_scale_student.py` | Triple-angle/codimension-two prototypes. Local-maximum/Jacobian checks with global uniqueness assumed; no validated selected-global joint-statistic reference. |
| Laplace location-scale | `loc_scale_laplace.py` | Standalone prototype; no static incoming import found. Preserve pending owner/domain review. Even-n midpoint preservation and nonsmooth conditional measure need a separate audit. |

No separate scale-only or general Gaussian sampler module was found among current model files. Gaussian controls exist in the affine/reference work. A future directory design should not imply nonexistent validated implementations.

### Reference, proof, diagnostics and execution infrastructure

- `kde_ref/moments.py` supplies raw weighted parameter summaries and KDE-derived summaries. `posterior.py` supplies smoothed density approximations; `reference_adapter.py` combines legacy tables and cached MLE errors.
- Direct **joint** weighted/rejection reference generation is spread over `followup_shifted_student_reference.py`, `validate_student_selected_mle_gibbs.py`, `validate_cheap_selected_mle_gibbs.py` and `validate_student_independence_mh.py`. An earlier implementation remains in `reporting/presentation/prepare_feedback_evidence.py`. These are active mathematical dependencies, not just disposable presentation builders.
- `scripts/selected_mle_reference_oracle.py` is the independent rational stationary-root/global comparison oracle. Keep it logically independent of production selection. `reporting/diagnostics/student_n5_boundary.py` provides a specialized exact algebraic boundary calculation; it is not a general algebraic-input production selector.
- `student_n3_selected_geometry.py`, `student_higher_n_geometry.py`, `student_n5_boundary.py` under `reporting/diagnostics/` contain reusable proof/geometry utilities. `verify_student_n3_components.py`, branch-component/transition studies, their drivers, archived plots and restart tables supply historical and current evidence.
- Drivers import one another: block validation imports selected-RATTLE starts/preflight and pair-Gibbs particles; several validations import summaries, bootstrap, ESS and oracle helpers from the shifted follow-up; n=3 fast-path validation imports geometry-driver helpers. Tests import these scripts and reporting utilities. Move them only after extracting/redirecting those actual dependencies.
- `analysis.py`, `scripts/run_cost_audit.py` and the recent follow-up diagnostics contain different ESS/summary implementations. Their definitions and failure behavior differ; consolidation must not retroactively replace historical metrics.
- `diagnostics/run_registry.py`, `configs/analysis_run_registry.yaml`, dashboard preparation and `run_analysis_pipeline.py` describe the older multi-family workflow. The latter can launch full reference/sampler jobs; it is not a harmless cleanup check.
- `configs/student_acceptance_k2_n10.json`, `student_rattle_k2_tuned_20260817.json`, timing/Numba configurations and `hpc/grace/` encode historical execution rules. Current selected studies mostly freeze configurations in `docs/research/evidence/<study>/config.json` rather than the root config directory.

### Scientific artifact groups to preserve

The study names below are literal paths, not proposed new copies. Each compact evidence directory carries its own schema; manifests use different keys (`raw_artifacts`, `raw`, `raw_arrays`, `artifacts`) and cannot safely be merged by assuming one common format.

| Compact evidence under `docs/research/evidence/` | Original raw/supporting location | Retention role |
|---|---|---|
| `selected_mle_reference_2026-10-03/` | `tmp/selected_mle_reference_validation_20261003/` | Initial independent oracle banks, controls and unfavorable shifted screen; dated selector generation |
| `selected_mle_selector_2026-10-10/` | Evidence-directory preserved original selector, replay tables and `tmp/selector_hardening_*` logs | Before/after correctness and performance; keep separate from the current selector |
| `shifted_student_followup_2026-10-10/` | `tmp/shifted_student_followup_20261010/` | Fresh weighted/rejection banks, six affine chains, source-at-generation and failure records |
| `student_selected_mle_gibbs_2026-10-10/` | `tmp/student_selected_mle_gibbs_20261010/` plus shared shifted banks | Transparent pair baseline, fresh n=3 reference and joint comparisons |
| `student_selected_mle_rattle_2026-10-10/` | `tmp/student_selected_mle_rattle_20261010/` plus shared references | Correctness tests and n=10 mixing failures; these failures are essential evidence |
| `cheap_selected_mle_gibbs_2026-10-10/` | `tmp/cheap_selected_mle_gibbs_20261010/` and `_20261010_v2/` | First narrow-proposal failure and repaired bounded study; the report discloses that a partial failing chain was not fully saved |
| `student_n3_geometry_selected_mle_2026-10-10/` | `tmp/student_n3_geometry_selected_mle_20261010/`; older n=3 component/transition directories in `results/` | Proof, graph, chambers, excursions, stress cases and historical reconciliation |
| `student_n3_fast_path_2026-10-10/` | `tmp/student_n3_fast_path_20261010/` | Frozen original block implementation, matched seeded trajectories, endpoint certificates and throughput |
| `student_independence_geometry_2026-10-10/` | `tmp/student_independence_geometry_20261010/` plus shared banks | Independence/block comparison and preliminary n=4/5 geometry in one study group |
| `student_n5_boundary_2026-10-10/` | Exact certificates/plot and initial-attempt snapshots in the evidence directory | Mathematical boundary theorem and selector audit; no raw MCMC experiment |

Older `results/student_n3_branch_components{,_clean,_safe}/`, the overlapping `student_mle_transition_grid_*` groups, `student_n3_discussion_note/`, final-production/cost/release-information studies and `docs/presentation/evidence/` remain historical evidence. Preserve distinctions among base, refined and remaining grids; pooling or selecting one as canonical needs the saved reconciliation, not a directory-name guess. `results/latest/` is a convenience view, not a stable provenance identifier.

## 3. Canonical versus historical implementations

“Canonical” should mean an explicit role and validated domain, not a blanket production endorsement. Keep the following authority chain, preserving dates and contrary findings:

| Authority | Current conclusion | Evidence scope |
|---|---|---|
| [Target audit](selected_mle_target_audit.md) | Score and selected-statistic measures differ; indicator alone is insufficient | Regular-region derivation; singular conditional laws not settled |
| [Reference validation](selected_mle_reference_validation.md) | Equivariant weighted/rejection identity and local affine ratio justified | Initial five-of-six screen; old selector/shifted limitations subsequently addressed in separate reports |
| [Selector hardening](selected_mle_selector_hardening.md) | Exact-vs-near ties, enumeration, flat roots and explicit failures improved | 157 checked selections; ordinary float search not universally certified; ambiguous exact fallback generally limited to n<=20 |
| [Shifted follow-up](shifted_student_reference_followup.md) | Fresh raw joint reference reliable for declared central and tail-6 summaries; local affine joint mixing fails | Two independent 65,536-particle banks, pooled importance ESS 14,747; oracle and independent rejection checks; rare upper parameter tail unresolved |
| [Pair Gibbs](student_selected_mle_gibbs_validation.md) | Transparent correction passes three fixed k=2 location cases | Parameter **and** latent checks; expensive; finite tests do not prove irreducibility |
| [Selected RATTLE](student_selected_mle_rattle_validation.md) | Regular endpoint invariant-target argument justified; both n=10 cases fail latent screens | n=3 small-run pass with weak tail precision; no universal numerical symplectic certification |
| [Cheap block Gibbs](cheap_selected_mle_gibbs.md) | Reversible regular score blocks with endpoint selection/J correction pass nine bounded settings | k=2, n=3/10, m=1/4/16; preserve first failed narrow-proposal run and v2 repair evidence |
| [n=3 geometry](student_n3_geometry_selected_mle.md) | Seven finite components, 25 chambers; central 13-chamber component is selected-MLE support | Exact proof for finite observations, normalized fixed-scale location; finite RATTLE projection is distinct from continuous motion |
| [n=3 fast path](student_n3_fast_path_validation.md) | Bracket certificate preserves matched tested chains and improves throughput | k=1/2/3, optional; 1.58–2.05x warmed speedup; one k=1,m=4 ESS screen fails; near-flat failure guards remain |
| [Independence MH](student_independence_mh_validation.md) | Simple justified E-coordinate alternative; two n=10 cases pass bounded screens | Centered efficient; shifted tail throughput worse than block Gibbs; no curvature factor in this representation |
| [Higher-n preliminary geometry](student_higher_n_geometry_preliminary.md) | Fold framework extends; even-n singularities and multiple maxima defeat n=3 shortcuts | Chamber adjacency counts are not a full n=4/5 topological classification |
| [n=5 boundary](student_n5_selection_boundary.md) | Unique global transition in one symmetric family inside a regular unchanged chamber | Exact theorem, triple global tie, lower-tie exclusion of zero; not classification of the whole chamber/manifold |

The n=3 theorem gives unique global selection and the median interval for any three finite observations. Its floating implementation additionally needs a reliable **root-location enclosure**, not only a small score residual. At n=5, normalized data `(-a,-a,0,a,a)` switch at the positive root of `256 A^5 - 1973 A^4 - 7292 A^3 - 4310 A^2 - 1100 A - 125`, with A=a² and a approximately 3.2495052. Zero remains a strict local maximum; exactly three global maxima coexist at the boundary and the smallest is selected. Therefore chamber identity or positive curvature cannot replace global selection at higher n.

Retain dated historical sources: `docs/student_n3_score_manifold_components.md`, `student_n3_gibbs_rattle_discussion.tex`, inverse-branch audit/follow-up, `affine_plan.md`, `docs/loc_scale.tex`, `docs/lit_review.tex`, presentation notes/decks and the draft `docs/paper/paper.tex`. Later add “superseded on these claims” links where appropriate; do not overwrite old experiment narratives or erase failed results. The September checkpoint and evidence ledger are historical anchors, not current validation summaries.

## 4. Correctness and documentation concerns

Severity **P0** means prevent misleading target/reference comparisons before any migration; **P1** means resolve or visibly constrain before a portable collaborator handoff; **P2** means organizational debt. These findings are recorded, not fixed.

### Target contracts to preserve

For unit-scale Student location, C=sum psi(x_i-t), G=sum psi'(x_i-t)², J=sum psi'(x_i-t). On regular regions:

\[
\rho_{score}\propto F_\mu/\sqrt G,
\qquad
\rho_{selected}\propto \mathbf1\{T(x)=t\}F_\mu |J|/\sqrt G.
\]

J is positive at a regular selected maximum. G is the score-manifold Gram quantity; J is the estimator-curvature correction. They are not alternatives. In equivariant base-data coordinates, the reference and independence-MH target is instead `f0(E) prior(t-T(E))`; no extra J or Gram factor belongs there. Gaussian constant curvature permits coincidence; Logistic uniqueness alone does not. Laplace requires its explicit median/order-statistic target.

| Priority/location | Confirmed issue or limited claim | Proposed treatment |
|---|---|---|
| **P0**, `models/model_registry.py:102–142` | `target_matches_reference=True` defaults for Student and Logistic score Gibbs/RATTLE. Model-level score description is also reused for raw-MLE/KDE method rows. | Method-specific target/selector/reference identities; mark unequal targets and prohibit rankings as same-target correctness comparisons. Preserve historical results with corrected annotations. |
| **P0**, same file Student convention | Says lower location for “numerical ties”; hardened selector deliberately removed this convention. | State smaller location only for proved exact global ties, with unresolved comparisons reported as errors. Version/fingerprint selector semantics independently of sampler target. |
| **P0**, `models/loc_student.py:372`, `loc_logistic.py:110`, `docs/gibbs_student.md:179–204` | Docstrings label latent conditioning as MLE; note explicitly asserts score preservation automatically preserves global MLE. | Correct documentation to actual score target. Do not change the legacy kernel to make the label true. |
| **P0**, `scripts/run_cost_audit.py` dispatch; targeted/final runners | “Production/final” runners invoke legacy score Student/Logistic modules. Reports can compare these to selected-MLE raw references. | Explicit method IDs and mismatched-target warning; retain this workflow as a historical score comparison. Do not silently substitute selected kernels/defaults. |
| **P0**, `kde_ref/reference_adapter.py:264–307` | `selected_global_v1` persists across selector hardening; existing z-sample cache loads without source/tolerance/shape/finite/provenance validation, even when cache writing is disabled. | Separate provenance-validation correctness task. Unknown caches remain retained but inadmissible as validated selected-MLE references until independently checked/regenerated. |
| **P1**, `reporting/diagnostics/audit_kde_reference.py:118–176` | Resume keys omit release/prior; filename/version and header checks cannot distinguish changed selector or posterior settings. | Manifest/run identity includes estimator generation and posterior settings. E-only banks may legitimately serve many priors; derived posterior summaries may not. |
| **P1**, `kde_ref/moments.py:182–217` | Gaussian KDE/Gaussian-prior helper loses original log-weight offset for normalization; quantiles use component means and omit within-component variance. Mean/variance formulas are separate and correct. | Isolated scientific bug fix with analytic mixture-CDF and predictive-normalization tests; exclude flawed fields from ground-truth status now. Not part of a file move. |
| **P1**, `configs/analysis_run_registry.yaml`, dashboard/reference consumers | All five runsets name absent `reporting/diagnostic_outputs/model_reference_audit/reference_all_models.csv`. | Show missing provenance explicitly. Do not fill the old path with unrelated fresh banks. New selected studies need their own reference IDs and domains. |
| **P1**, `README.md`, `docs/roadmap.md`, `docs/evidence_ledger.csv`, `docs/paper/paper.tex` | Affine validation and exact-location milestone still described as pending; ledger geometry/selector statuses predate proofs/hardening. | Add current dated status and links. Keep old baseline claims dated; use an evidence-ledger revision/supersession chain, not rewritten history. |
| **P1**, `docs/student_n3_gibbs_rattle_discussion.tex:90–170` | Continuous-path restrictions are applied to finite numerical RATTLE proposals. Older component note lacks displayed proof now supplied elsewhere. | Link new proof and finite-projection counterexample; separate exact continuous geometry from finite map behavior and observed transition counts. |
| **P1**, affine/RATTLE reports and old presentations | Parameter agreement/acceptance alone does not establish joint exploration. Shifted affine and n=10 RATTLE failed latent screens. | Keep explicit validation/mixing status per case; never promote them to joint reference or infer zero rare-event uncertainty. |
| **P1**, joint Student modules/`docs/loc_scale.tex` | Local-maximum checks with assumed uniqueness are not selected-global joint selection. Singular estimator configurations remain outside regular derivations. | Preserve assumption-dependent research classification; dedicated estimator/reference work must precede exact joint claims. |
| **P1**, `loc_scale_laplace.py` | Claims exact joint-MLE target, while even-n fixed-side moves do not by themselves preserve the midpoint median; target-measure audit absent. | Preserve as unvalidated prototype. Audit constraints/coarea and estimator convention separately before use or promotion. |
| **P1**, Laplace location/facet code | Fixed label/facet cannot be described as a globally exchangeable latent sampler; even-n interval target is a different event/statistic. | Retain chart restrictions and odd/even metadata; verify intended symmetric summaries separately from joint labeled laws. |
| **P1**, numerical guards across samplers | Legacy EPS cutoffs, newer regular chart exceptions, near-flat endpoint guards and selector absolute tolerance have distinct contracts. | Record them; do not homogenize guards during cleanup. A selector error is not an ordinary target rejection or a license to discard reference particles. |
| **P2**, `analysis.py`, cost scripts, follow-up diagnostics | Duplicate ESS implementations, including constant-chain and nonfinite-data handling differences. | Named metric definitions and centralized *new* analysis after independent tests; preserve historical outputs and frozen criteria. |
| **P2**, `models/__init__.py`, root imports, script-to-script imports | Eager imports, mixed responsibilities, private-helper cross-module dependencies. | Stage package/helpers gradually with compatibility tests; no wholesale rewrite. |
| **P1/P2**, `results/README.md` and old artifact references | Names absent `_archive_repo_cleanup_20260608/`; it cannot substantiate claimed recovered contents. Paper figures depend on presentation assets. | Record absence, trace/retain original figure assets and scripts; do not assume a newer checkpoint recovers files already absent at capture. |

No newly established branch-Jacobian defect is claimed in the legacy Student pair mathematics. The inverse-Jacobian branch weights are already correct. However, the cheap-Gibbs report shows why its literal cutoffs/approximate sampling cannot simply be wrapped in an exact endpoint W-ratio. The separate regular compiled block deliberately avoids those issues. Similarly, selected RATTLE shares integration machinery but adds phase admissibility/reversal checks; swapping its wrapper for legacy checks would change its numerical contract.

No module is declared disposable from search absence. `loc_scale_laplace.py` is a candidate for experimental quarantine, not deletion. Static checks found no unresolved explicit local imports; dynamic loaders, optional imports, missing artifacts and environment-dependent runtime behavior remain separate risks.

## 5. Proposed future directory structure

Use a small installable package and explicit method names. This is a destination, not permission to move everything in one change. Avoid a plugin/framework hierarchy; the current parameter dictionaries can remain during migration.

```text
pyproject.toml                       # package metadata; no dependency upgrades
src/student_t_mle/
  models/                           # densities, scores, derivatives; known-scale first
  estimators/                       # selected location MLE, errors and conventions
  targets/                          # score/selected definitions and endpoint checks
  samplers/
    student_score_gibbs.py
    student_score_rattle.py
    student_selected_pair.py
    student_selected_block.py
    student_selected_rattle.py
    student_selected_independence.py
    affine_location.py
    logistic_score_gibbs.py
    logistic_score_rattle.py
    laplace_median_gibbs.py
    _student_mu.py                  # compact tested numerical helpers
    _numba_math.py
  geometry/                         # reusable chamber/algebra functions
  reference/                        # raw weighted/rejection identity and bank contracts
  diagnostics/                      # metrics, constraints, cost and provenance
experiments/
  validation/                       # bounded experiment orchestration
  geometry/                         # proof computation and figure drivers
  benchmarks/                       # frozen cost studies
  experimental/                     # joint models and facet RATTLE, initially unchanged
tests/                              # focused tests; small deterministic fixtures
docs/
  research/                         # canonical reports, their compact evidence unchanged
  paper/                            # manuscript and separately traced figure inputs
  checkpoints/                      # immutable historical checkpoints
  archive/                          # indexed historical notes/workflows, after review
evidence/                           # optional future manifest/index, not duplicate banks
models/, kde_ref/, scripts/, reporting/  # transitional compatibility paths
```

Keep bulky arrays in an explicitly chosen external artifact root with a manifest mapping original paths to immutable storage locations. **No artifact-root environment variable or resolver currently implements this proposal.** Do not require new storage migration to begin the first documentation phase. Historical evidence snapshots, configurations and manifests remain at original paths until a separate compatibility plan is verified.

### File-level migration map

| Current files | Proposed destination | Required compatibility/dependency arrangements |
|---|---|---|
| `models/student_location_mle.py` | `estimators/student_location.py` | Preserve `global_mle`, `stationary_points`, diagnostic functions, errors, tolerances and `get_mle` wrappers; independent oracle remains separate. |
| `models/student_n3_endpoint.py` | `targets/student_n3_endpoint.py` | Preserve optional/default behavior, interval rounding, diagnostic types; no fastmath/reassociation. |
| `geometry`, `selected_state` in selected-RATTLE module | `targets/student_location.py` | Extract only in its own reviewed step; old module re-exports helpers used by block and tests. |
| `models/loc_student.py` | `samplers/student_score_gibbs.py`; later `models/student.py` for pure model functions | Initially move as a unit with old-path shim. Split mathematics only after replay, avoiding simultaneous code deduplication. |
| `models/loc_student_rattle.py` | `samplers/student_score_rattle.py` | Preserve compiled integrator/private helpers consumed by selected RATTLE and audits. |
| `models/loc_student_selected_mle.py` | `samplers/student_selected_pair.py` | Preserve joint proposal/branch law and result diagnostics; retain correctness baseline. |
| `models/loc_student_selected_mle_block.py` | `samplers/student_selected_block.py` | Keep m, RNG streams, selection order, failure telemetry and n3 option identical. |
| `models/loc_student_selected_mle_rattle.py` | `samplers/student_selected_rattle.py` | Keep identity mass, Gram force, full phase reversal and support checks. |
| `models/loc_student_selected_mle_independence.py` | `samplers/student_selected_independence.py` | Preserve E-state caching, once-per-proposal selection and translation failure guard. |
| `models/loc_affine.py` | `samplers/affine_location.py` | Keep `run_gibbs` alias; expose a descriptive name only additively; preserve family/scale argument conventions. |
| `student_mu_mh.py`, `numba_math.py` | `samplers/_student_mu.py`, `_numba_math.py` | Preserve compiled arithmetic and seeding; shims for existing private imports. |
| Logistic/Laplace location plus compiled helpers | Named comparator sampler modules, model math later | Preserve odd/even target and fixed-label semantics, API shapes and JAX configuration behavior. |
| `loc_scale_student.py`, `loc_scale_student_rattle.py`, `loc_scale_laplace.py`, `loc_laplace_rattle.py` | `experiments/experimental/` initially or a later experimental package | Keep original import wrappers, dedicated tests and assumption metadata; do not advertise as validated general selected-MLE implementations. |
| `kde_ref/moments.py`, `posterior.py`, `reference_adapter.py`, `numba_kernels.py` | `reference/weighted.py`, `reference/kde.py`, `reference/banks.py` and compact kernels | Correctness changes to bad KDE fields/cache identity are separate; preserve historical entry points and output schemas. |
| `scripts/selected_mle_reference_oracle.py` | `experiments/validation/oracles/student_rational.py` | No dependence on production root selection; compatibility module for tests/old commands. |
| `reporting/diagnostics/student_n3_selected_geometry.py`, `student_higher_n_geometry.py`, `student_n5_boundary.py`, `verify_student_n3_components.py` | `geometry/` pure helpers; execution stays in geometry experiments | Preserve exact symbolic identities and chamber labels; distinguish graph enumeration from topology. |
| Recent `validate_*`, shifted follow-up, `study_student_*`, `audit_student_n3_fast_path.py`, evidence verifiers | `experiments/validation/` or `experiments/geometry/` | First extract actively imported summary/particle helpers; old CLI wrappers preserve flags and safe output behavior. |
| `diagnostics/{run_registry,cost_ledger}.py`, `analysis.py` | `diagnostics/` with named legacy metrics retained | Explicit target-aware joins; no metric-definition changes disguised as relocation. |
| Older production/tuning/profile runners, `hpc/grace/`, configs | `experiments/benchmarks/` or indexed historical experiments | Update actual job commands with wrappers; old “final” IDs stay historical, not selected-MLE aliases. |
| Dashboard and presentation builders/assets | Later `docs/archive/` / indexed historical tooling | Keep numerical reference helpers needed elsewhere; retain paper figure paths or compatible mapping before moving assets. |
| Research reports/evidence and original `results/`, `tmp/` banks | Keep current report paths; later manifest-directed artifact storage | No rewritten source hashes, erased failures, regenerated artifacts under old identities or automatic removal. |

The package may initially retain family-and-algorithm modules rather than immediately splitting every density function. This avoids circular imports and an abstraction layer that is larger than the research problem. Target names and validation status belong in explicit method metadata; they should not be inferred from `run_gibbs`, `get_mle`, file names or a model-level registry.

## 6. Reproducibility and preservation requirements

### Minimum preservation contract

The existing [September preservation package](../checkpoints/2026-09-20/preservation.md) has a verified historical restore drill, including file modes, Git bundle and patches. It predates October research and lives on the same filesystem. Its existence is not protection for all current work or an offsite backup. Before moves, capture current Git refs/status, staged/unstaged binary patches, new source and evidence manifests, and recovery instructions **outside this checkout** or in a reviewed incremental store. Do not invoke `preserve_repository.py` blindly: its documentation warns that an earlier nested checkpoint can be included in a subsequent snapshot.

Preserve the following separately:

1. Source, uncommitted distinctions, configs, seeds, algorithm versions, environment/CPU/thread metadata, tolerance and selection conventions.
2. Original arrays for validated references and saved chains, rejected/failed-run records and unfavorable comparisons. Compact summaries alone cannot reconstruct joint latent diagnostics or exact seeded trajectories.
3. Exact proof certificates, rational isolations, graph/chamber labels, source-at-generation copies, manifest and figure provenance.
4. Editable presentation/manuscript sources and externally sourced assets where originals have independent value. The paper currently reads `docs/presentation/Images/final/`; the n=3 discussion reads generated PDFs in `results/student_n3_discussion_note/figures/`.

Bytecode/Numba caches, installed environments, generated density grids, LaTeX intermediates and dashboard views are potentially rebuildable. **Classification as rebuildable does not authorize deletion.** A density grid backed only by an unknown old MLE bank is not a trustworthy replacement reference. Failed execution logs and intermediate source snapshots can be scientifically necessary even when later outputs exist.

The three checked reference files are:

| Original path | Bytes | Verified SHA-256 |
|---|---:|---|
| `tmp/student_selected_mle_gibbs_20261010/fresh_n3.npz` | 1,012,535 | `e6d99316712f1a11ec7ea4237b73ccc49db691fc3d679d96f900a98998a17f02` |
| `tmp/shifted_student_followup_20261010/weighted_a.npz` | 5,570,774 | `000fc3e8ecbf31088a15d853749af2cd7768febd8b316f924f81aff101b261e6` |
| `tmp/shifted_student_followup_20261010/weighted_b.npz` | 5,571,133 | `b2f5b0197a6c19fae1f2775e2de33901746b319ede6167e8e2751088ee376c35` |

Bank identity must include base distribution, k/n/scale, generator/RNG/seed, selected statistic and tie convention, selector source hash/tolerances, failure/fallback counts and array schema. Release/prior identify the derived weighted posterior, not necessarily the reusable standardized bank. Add figure/report edges linking claim -> code/configuration -> original bank/chain -> analysis/version -> output. Keep unknown historical provenance explicitly unknown.

### Regression gates, without rerunning full studies

| Protected property | Existing focused checks / necessary migration check |
|---|---|
| Global selection, ties, flat/extreme roots and failure propagation | `test_student_location_mle_hardening.py`, `test_student_global_mle_audit.py`, `test_selected_mle_reference_validation.py`; replay the saved adversarial oracle subset, including near/exact ties and output-precision failures |
| Pair transformation/Jacobians and legacy numerics | `test_student_inverse_branch_weights.py`, `test_student_gibbs_backends.py`, `test_student_rattle_diagnostics.py`; unchanged-trajectory replay in fixed environment |
| Selected pair/block detailed balance and constraints | `test_student_selected_mle_gibbs.py`, `test_student_selected_mle_block.py`; compare fixed initial states, uniforms, accept decisions, diagnostics and retained datasets |
| Guarded n=3 support certificate | `test_student_n3_endpoint.py`, graph/selected-geometry tests; all chamber witnesses, inward/outward brackets, near-flat/fold/large-offset cases; match option-on/off seeded decisions where justified |
| Selected RATTLE phase geometry | `test_student_selected_mle_rattle.py`; score/tangent constraints, momentum reversal, full phase return, endpoint energy and explicit selector errors |
| Independence/affine identities and reference construction | `test_student_independence_mh.py`, `test_loc_affine.py`, `test_shifted_student_followup.py`, raw weighted tests; preserve caches/RNG and valid translation reconstruction |
| Comparator/joint contracts | `test_model_registry_and_location_conventions.py`, `test_logistic_rattle_diagnostics.py`, `test_laplace_reference_targets.py`, both facet suites, `test_loc_scale_student.py`; keep non-global joint status visible |
| Proof code | n=3 component/selected-geometry, `test_student_higher_n_geometry.py`, `test_student_n5_boundary.py`; exact polynomial/resultant/Sturm certificates and chamber graph agree |
| Metadata/cache/reporting correctness | Current registry/reference tests plus new focused method-target mismatch, cache fingerprint, resumability and KDE mixture-CDF tests in separate changes |
| Existing evidence portability | Hash original banks/chains against manifests; regenerate selected **small** plots/tables into a new destination from saved data; document font/rendering differences |

For pure organizational moves on the same environment, require bit-identical short seeded arrays and accept decisions where arithmetic/RNG order is unchanged. Record Numba/JAX cache state; cold compilation is not an algorithm speed regression. Cross-platform or intentional arithmetic changes require constraint/root error checks and bounded statistical comparisons with reference uncertainty, not a claim of identical floating trajectories. Keep scientific changes separate so this distinction is reviewable.

Use scoped `.venv/bin/python -m pytest tests ...` discovery; do not collect temporary third-party tests. Set thread/CPU conventions, disable unrelated plugin autoload where appropriate, record collection/timeout failures, and use fresh output paths. The September **113-test** baseline and later report-specific passes are historical; this audit establishes no new whole-suite pass. A scoped full-suite baseline should precede packaging, after focused checks and environment availability are confirmed. No expensive chains or broad grids are required for routine relocation gates; existing arrays support most comparisons.

### Environment requirements

The current interpreter is Python 3.13.5. Metadata reports NumPy 2.5.2, SciPy 1.18.0, Numba 0.67.0, JAX/JAXlib 0.11.0, pandas 3.0.5, Matplotlib 3.11.1 and pytest 9.1.1. These are inspected installed versions, not a recommended upgrade set. Root requirements mix numerical, test, plotting and Streamlit dependencies and are loosely bounded. `np.trapezoid` in current code is incompatible with the advertised NumPy >=1.20 range; tightening a declared minimum requires a separate compatibility review.

SymPy is declared but absent from `.venv`; prior oracle/hardening runs use the separate `/tmp/student-mle-oracle-deps-20261003` installation. A fresh collaborator environment must provide SymPy for ambiguous selectors and algebraic tests. Optional statsmodels is absent; KDE has a fallback whose use must be recorded. pyarrow is installed for parquet; optional presentation dependencies are listed in `reporting/presentation/requirements.txt`. Capture a fresh freeze and distinguish core, testing/oracle, plotting and historical UI/deck requirements without upgrading them during structural cleanup. Fresh installation is an explicit future check, not an established capability.

## 7. Ordered cleanup phases

Each phase is an independently reviewable change. No phase below is executed by this report. Rollback means reverse **only that phase's delta** in an isolated checkout or restore it from an external patch; never `git reset --hard`, blanket `git clean` or restoration over user-owned work.

### Phase 1 — Current research entry point

- **Files:** `README.md` only in the first change; then a separate change to `docs/roadmap.md` and `docs/evidence_ledger.csv`.
- **Change/reason:** replace the stale current-status paragraph with method-specific status and links to authoritative reports. Add current known-scale selected-MLE path while retaining the explicit score-conditioned example and dated baseline. Later record superseded ledger statuses rather than losing their history.
- **Dependencies/risk:** none beyond this reviewed blueprint; low risk, but overclaiming exactness/mixing is a scientific risk.
- **Verification:** link resolution and statement-by-statement comparison to canonical reports; source/default and original artifact hashes unchanged.
- **Rollback/completion:** restore only the documentation patch; complete when a reader can identify the two targets, the tested selected-MLE implementations and failing/assumption-dependent cases without reading old runners.

### Phase 2 — Method and reference metadata

- **Files:** `models/model_registry.py`, relevant consumers in `reporting/diagnostics/audit_reference_all_models.py`, `scripts/run_cost_audit.py`, `diagnostics/run_registry.py`, dashboard audit/cache consumers, registry tests. Separate documentation patch for legacy model docstrings and `docs/gibbs_student.md`.
- **Change/reason:** per-method target IDs, selected-statistic/tie convention, reference ID, domain and validation status; block invalid same-target comparisons. Correct Student numerical-tie wording and raw-reference descriptions. Existing runner dispatch stays unchanged.
- **Dependencies/risk:** Phase 1; CSV schemas and dashboard assumptions may break. This is metadata/reporting correctness work, not a kernel change.
- **Verification:** registry/target tests, fixture-based reference joins, historical output read compatibility, no sampler hash/RNG changes. New summaries must identify missing/mismatched references.
- **Rollback/completion:** keep old schema reader or explicit conversion; reverse phase-specific patch. Complete when no registered score method can be silently labeled as matching a selected-MLE reference.

### Phase 3 — Recovery and portable evidence index

- **Files:** existing checkpoint tools/docs reviewed; add a current manifest/evidence index, paths for the ten `docs/research/evidence/` study groups, original `tmp/`/`results/` dependencies and figure edges. No array moves yet.
- **Change/reason:** record current protected working-tree changes and ignored scientific artifacts externally; choose external immutable storage destination. Reconcile the missing older archive/reference table and unknown legacy provenance.
- **Dependencies/risk:** before any module or artifact moves; metadata can start alongside Phase 2. Risk of recursively backing up the old 9 GB checkpoint or misclassifying tmp evidence.
- **Verification:** current source/evidence hashes, three bank hashes above plus all planned-to-move artifacts; isolated restore drill including modes and staged/unstaged distinctions. No storage deletion.
- **Rollback/completion:** new external manifests/checkpoints are additive; restore mappings in a separate destination. Complete when every proposed move has recovery and report/figure dependency entries. Do not treat the old checkpoint as sufficient for October work.

### Phase 4 — Package scaffold and import boundary

- **Files:** new `pyproject.toml` and package scaffold; `models/__init__.py` only in a distinct reviewed lazy-import change; setup/example instructions; packaging/import smoke tests.
- **Change/reason:** make installation and root-independent imports possible before distributing code across directories. Declare currently necessary dependencies with no upgrades. Capture the tested environment first.
- **Dependencies/risk:** Phases 1–3 and a scoped baseline; import-time JAX settings, package discovery and compiled-cache behavior are real risks.
- **Verification:** isolated editable/install smoke, old/new import names, example shapes/target, scoped tests. Check that importing selector/target helpers does not accidentally change global configuration or RNG state.
- **Rollback/completion:** remove only new scaffold/install metadata or reverse lazy-import patch; retain original checkout untouched. Complete with both import paths supported and documented environment limits.

### Phase 5 — Numerical module migration, one dependency group at a time

- **Files:** selectors/shared math first; target helpers/n3 certificate next; then Student score, selected pair/block/RATTLE/independence/affine modules and comparator modules according to the migration table. Keep wrappers in `models/`.
- **Change/reason:** expose explicit target names, remove dependence of block target checks on a RATTLE namespace, separate model/estimator/sampler responsibilities gradually.
- **Dependencies/risk:** Phases 3–4; no force, proposal, branch, tolerance, seeding, acceptance-order or default changes. Private helper imports, Numba globals and dataclass identities can break apparently simple moves.
- **Verification:** focused gates for each group, fixed-environment seeded replay, selected-state/error/diagnostic equivalence, shared-oracle comparison. Run scoped full-suite regression after all groups, not a broad new experiment.
- **Rollback/completion:** one group per patch, wrappers delegate to exactly one implementation; rollback with original module hashes. Complete when old commands and new imports work, algorithm identity and validation status are unchanged, and no duplicated implementation has divergent fixes.

### Phase 6 — Reference and diagnostic extraction

- **Files:** `kde_ref/`, common helpers currently imported from shifted/Gibbs validation drivers, `analysis.py`, `diagnostics/`, tests.
- **Change/reason:** raw joint reference generation belongs outside experiment orchestration; named diagnostic definitions avoid accidental metric substitution. Keep oracle separate. Cache-provenance/KDE-quantile fixes must each have their own scientific correctness patch and tests **before affected routines are endorsed**, not be smuggled into extraction.
- **Dependencies/risk:** metadata contract and preservation first; package migration can proceed for unaffected components. Risk of circular driver dependencies, cache aliasing and revised quantiles/ESS changing historical reports.
- **Verification:** same saved particles/weights/raw quantiles and bootstrap inputs; analytic Gaussian and mixture checks; bank fingerprint/failure tests; no inherited unknown cache becomes trusted merely by loading.
- **Rollback/completion:** driver compatibility exports remain; retain old historical diagnostic definitions/output identities. Complete with one documented raw reference API and explicit independent oracle, KDE and legacy-metric roles.

### Phase 7 — Experiment entrypoints and indexed historical workflows

- **Files:** recent validation/proof/benchmark drivers, their imported helpers, relevant `hpc/grace/` commands/configurations; later dashboard/presentation tooling and stale notes.
- **Change/reason:** bounded reproducible commands outside kernels; reduce duplicated orchestration. Archive only after numerical helpers and paper/figure dependencies are accounted for.
- **Dependencies/risk:** Phase 6 extraction before moving scripts/reporting; file-relative ROOT calculations, dynamic loading of frozen source, TSV jobs and literal artifact paths are fragile. Evidence snapshots remain immutable.
- **Verification:** CLI/config parsing with no job launch, small saved-data replay, output-directory refusal/safety, figure/source-path audit, old wrapper commands. Review any archive candidate with no callers rather than declaring it unused.
- **Rollback/completion:** redirect/reverse only reviewed file mappings, keep provenance/index and compatibility links. Complete when every active command has a target, budget, config, evidence path and stable entrypoint, and archives can be reconstructed.

### Phase 8 — Artifact portability and collaborator/manuscript handoff

- **Files:** approved original banks/chains/figures, manifest/resolver, README/report index, `docs/paper/` figure references; no blanket tmp/results removal.
- **Change/reason:** separate bulky preserved evidence from normal source checkout and prepare the current known-scale scientific narrative.
- **Dependencies/risk:** all moved artifacts indexed and recoverable from Phase 3; manuscript work follows metadata/research review. Shared evidence across multiple reports can be severed by renaming; historical source hashes cannot be recomputed as if new code generated old outputs.
- **Verification:** destination hashes, restore/read tests, manifest-to-report graph, reproduction from saved arrays into fresh output, collaborator installation/example. No same-machine historical/fresh timing claims.
- **Rollback/completion:** originals retained until independently verified recovery and explicit disposition decision; restore old paths/mapping. Complete when the collaborator package includes assumptions, successful and failed validations, theorem scope, compact tables and access to immutable raw evidence.

## 8. Risks, dependencies and decisions requiring review

The following need review before their implementation phase; they do not prevent Phase 1 documentation:

1. **Canonical interface:** agree method IDs and metadata fields without changing existing function arguments or parameter dictionaries. Explicitly distinguish selected pair, block, endpoint RATTLE, local affine and independence MH.
2. **Compatibility duration:** keep old imports/CLI wrappers at least through collaborator migration and frozen-evidence replay. Private compiled-helper imports also require temporary compatibility; a simplistic `__all__` re-export may not preserve them.
3. **Storage:** choose an external artifact destination, ownership, durability and restore policy. Same-filesystem copies are recovery conveniences, not offsite protection. No new giant archive inside this checkout.
4. **Current evidence authority:** raw joint weighted references plus uncertainty/provenance are canonical numerical benchmarks on tested domains. Rejection is independent posterior sampling conditional on estimator accuracy; independence MH/block are efficient computational alternatives. RATTLE/affine cannot become ground truth on cases where latent diagnostics fail.
5. **Scientific fixes:** cache identities, bad KDE mixture quantiles/normalization, Laplace joint target, metric corner cases and general selector certification deserve separate changes. Organizational cleanup must neither alter nor conceal them.
6. **Experimental placement:** preserve Student joint and Laplace facet/joint code, notes/tests and future scale work without importing them eagerly into the known-scale core. Decide their eventual package location after the validated location interface is stable.
7. **Reproducibility claims:** loose dependency ranges do not constitute a portable tested environment. Oracle availability, IEEE rounding assumptions, finite root tolerances and unavailable old artifacts need visible contracts.

**Explicitly out of cleanup scope for now:** tuning Gibbs or RATTLE, making the n3 option default, changing parameter MH guards, n>=10 stress grids, RATTLE tail bridges, general n=4/5 topology classification, exact algebraic-input selector development, location-scale global selection, KDE optimization, large chain reruns, deleting historical/untracked/ignored material, dependency upgrades or automatic commits. The n=5 theorem does not authorize a chamber-level selector shortcut.

## 9. Recommended first implementation task and prioritized checklist

**The smallest safe first operation is a README-only status update.** Replace its stale affine-pending paragraph with a short current table linking target audit, hardened selector, fresh weighted references, selected pair/block/n3 option, independence MH and RATTLE/affine limitations. Retain the legacy example's explicit score target and historical test date. No imports, registry behavior, defaults, evidence or files move. Acceptance: all links resolve, claims match reports, and every non-README source/artifact remains unchanged. This is proposed for review, not started.

- [ ] Review this blueprint and approve the README-only phase.
- [ ] Refresh current roadmap/ledger with dated supersession links.
- [ ] Correct method/reference metadata and contradictory target labels; preserve old dispatch.
- [ ] Establish current external recovery/evidence manifest, including ignored banks and October untracked sources.
- [ ] Freeze a reproducible environment and run scoped baseline gates before packaging.
- [ ] Add minimal package/compatibility scaffold; migrate one dependency group at a time.
- [ ] Extract raw references/diagnostics with independent oracle and explicit metric contracts; fix scientific defects separately.
- [ ] Move active drivers only after helper dependencies are resolved; index historical workflows and figure assets.
- [ ] Verify external artifact recovery and saved-data regeneration before any future storage disposition.
- [ ] Assemble collaborator/manuscript package with bounded validation claims and remaining limitations intact.

Work stops at this blueprint. No cleanup phase, sampler change or new experiment is initiated.
