# Repository deep cleanup and deletion audit

Date: 2026-10-11. Scope: the entire current `student_t_mle` checkout, including ignored, untracked, hidden, recovery, environment and Git directories. **Preparation only: nothing has been deleted, moved or renamed.** Only this report and the accompanying CSV are added.

The revised objective is a lean working research repository, not indefinite retention of event materials or every generated result. This audit supersedes the blueprint's blanket recommendation to preserve presentation sources and all release-information exports. Unique historical numerical evidence can leave the checkout; redundant exports can be permanently discarded. Current selected-MLE mathematics, algorithms, independent references, failures, validation trajectories and manuscript inputs remain protected.

## 1. Current size and complete inspection scope

The inventory recursively used `lstat`/directory enumeration without excluding any directory or following symlinks. It recorded relative paths, kinds, logical bytes, allocated blocks, inode/device, modes, modification times and symlink targets. There were no unreadable paths. Credential/tool directory contents were not opened; the three managed roots are empty directories and excluded from proposed removal.

**Current logical size: 20,588,186,325 bytes (19.174 GiB). Current allocated size: 20,723,113,984 bytes (19.300 GiB).** There are 44,670 regular files, 4 symlinks and 5,587 directories, 50,261 total entries. No multiply linked regular files were found; all four symlinks are in the retained environment. The initial scan took 522.8 seconds. These sizes precede the two new audit documents.

| Root | Regular files | Logical bytes | Allocated GiB | Role |
| --- | --- | --- | --- | --- |
| `_archive_repo_cleanup_20260920_checkpoint` | 11 | 9,051,650,359 | 8.430 | Old recovery package |
| `results` | 16010 | 7,909,430,414 | 7.419 | Original/derived historical studies |
| `.venv` | 21301 | 1,847,864,888 | 1.775 | Current working environment |
| `.git` | 737 | 789,573,708 | 0.738 | History/index/refs; no GC proposed |
| `tmp` | 4883 | 816,710,827 | 0.772 | Current scientific evidence plus obsolete event work/vendors |
| `docs` | 804 | 150,107,782 | 0.142 | Research, manuscript and retired presentations |
| `models` | 566 | 15,722,811 | 0.016 | Protected numerics plus disposable JIT cache |
| `reporting` | 121 | 2,501,718 | 0.003 | Protected scientific helpers plus retired builders |
| `scripts` | 77 | 1,081,332 | 0.001 | Current validations and historical drivers |
| `tests` | 80 | 753,272 | 0.001 | Protected tests plus caches |
| `dashboard` | 15 | 201,331 | 0.000 | Retired Streamlit workflow |

The earlier blueprint excluded caches and temporary dependency names from some counts. In particular, `tmp/` is 816,710,827 logical bytes now, not the earlier restricted 751,120,423-byte count. This is a coverage difference, not evidence of new experiments.

Starting HEAD is `b4f66744158d6517e1cb5bfb628f770eb1ead743`: 165 tracked files, 32 modified tracked files and 723 untracked files. All 888 tracked/untracked regular files were hashed before inspection. Modified tracked sources remain KEEP, including the existing README and numerical work. No applicable checkout/ancestor `AGENTS.md` or root `MEMORY.md` was identified. The blueprint, checkpoint documentation and canonical research reports supply the current evidence map; personal memory only located historical preservation guidance, which was rechecked locally.

## 2. Classification and deletion-manifest contract

| Category | Units | Regular files | Logical bytes | Allocated GiB | Meaning |
| --- | --- | --- | --- | --- | --- |
| KEEP | 710 | 23229 | 2,839,941,463 | 2.704 | Active research/environment/history/manuscript; retain |
| DELETE | 172 | 10050 | 2,474,534,631 | 2.326 | Obsolete/redundant; remove after the stated prerequisites |
| ARCHIVE | 140 | 11389 | 6,565,331,557 | 6.150 | Unique historical evidence/reproduction source; verify external copy before local removal |
| REVIEW | 5 | 2 | 8,708,378,674 | 8.110 | Unresolved helper/recovery/managed directories; no automatic deletion |

A DELETE classification means scientific obsolescence/redundancy is established for the specified unit. It does **not** waive a stated dependency, input-preservation or consumer-retirement prerequisite. Wave 1 is the immediately reviewable low-risk subset. ARCHIVE is removal from the working checkout after verification, not another nested backup.

The CSV covers every existing non-directory path once, with no overlaps. Each row contains the requested classification, reason, significance, dependencies, reproducibility, action, prerequisites and savings, plus a unique ID, wave, file count, allocated size and membership fingerprint. There are 1027 units. Scope interpretation is mandatory:

- `exact single path`: precisely that file or symlink.
- `entire recursive directory; do not follow symlinks`: every descendant present in the inventory. Re-enumerate before removal and stop if new files appear.
- `only exact non-directory paths in members_json`: remove/archive **only** that JSON list. Explicit member lists are split into at most 700 paths for ordinary CSV readers. Several rows share the same release-study parent path because they have different categories. Never apply a recursive delete to those parent paths.

`scope_members_sha256` hashes the sorted non-directory relative paths joined by newline, with no trailing newline. It detects scope changes, not changes to file content. `size_bytes` is logical file/symlink size; `allocated_bytes` sums filesystem blocks for the unit. Reclaim estimates equal allocated size for DELETE/ARCHIVE; KEEP and REVIEW have zero automatic reclaim. The large REVIEW tar has a separate conditional future reclaim value. Units do not account for 9,994,240 bytes of unassigned parent-directory metadata, so sums of reclaim estimates are conservative relative to the full filesystem inventory. Quota snapshots/open handles may delay actual reclamation. No compression ratio has been assumed.

`known_dependencies` records exact textual/import references among inspected code/configuration/docs, supplemented by the manual traces below. Historical bookkeeping references and source-at-generation copies are not live importers. An empty search is not proof of disposability: dynamic loaders, runset paths, job commands, LaTeX includes and manifest dependencies were also inspected. Scientific files are not classified by Git status alone.

CSV SHA-256: `9f71d83505dd63e252ffa7c62db81a2f4e0eb1c7f027e1e250b97c0ba8fe2d65`. Treat the CSV as a review manifest, not shell commands.

## 3. Protected active research

KEEP covers all current models and tests, including score comparators; the corrected pair/block/RATTLE and independence samplers; optional guarded n=3 endpoint; affine sampler; hardened selector; shared compiled kernels and parameter update; independent rational/algebraic oracles; mathematical reports and proof certificates. Logistic/Laplace and location-scale work remain preserved. Cleanup must not turn a score comparator into a selected-MLE algorithm by renaming its output.

The following ignored raw study roots are KEEP, alongside their compact `docs/research/evidence/` packages:

```
tmp/shifted_student_followup_20261010/
tmp/student_selected_mle_gibbs_20261010/
tmp/student_selected_mle_rattle_20261010/
tmp/cheap_selected_mle_gibbs_20261010_v2/
tmp/student_n3_geometry_selected_mle_20261010/
tmp/student_n3_fast_path_20261010/
tmp/student_independence_geometry_20261010/
```

The first failed cheap-block study `tmp/cheap_selected_mle_gibbs_20261010/` and initial `tmp/selected_mle_reference_validation_20261003/` may move to verified external historical storage. Their failure/original-selector provenance remains distinct from later validated evidence. Do not erase unfavorable results or write regenerated arrays under an old bank identity.

Older n=3 dependencies still retained locally are `results/student_n3_branch_components_clean/`, `results/student_mle_transition_grid_n3_refined/` and `results/student_n3_discussion_note/`. Modern geometry also consumes checkpoint restart tables. Laplace facet validation/behavior results are retained. Broader old transition studies can be archived, but their differing grids, seeds and targets must not be merged merely because filenames resemble one another.

Fresh raw reference hashes were independently rechecked in this audit:

| Bank | Bytes | SHA-256 |
| --- | --- | --- |
| `tmp/student_selected_mle_gibbs_20261010/fresh_n3.npz` | 1012535 | `e6d99316712f1a11ec7ea4237b73ccc49db691fc3d679d96f900a98998a17f02` |
| `tmp/shifted_student_followup_20261010/weighted_a.npz` | 5570774 | `000fc3e8ecbf31088a15d853749af2cd7768febd8b316f924f81aff101b261e6` |
| `tmp/shifted_student_followup_20261010/weighted_b.npz` | 5571133 | `b2f5b0197a6c19fae1f2775e2de33901746b319ede6167e8e2751088ee376c35` |

These agree with the canonical evidence. This is not a new statistical validation experiment. Keep the standardized arrays, latent reconstructions, selector-generation metadata, bootstrap seeds, failures and chain inputs needed for joint/tail diagnostics. Compact parameter means alone cannot replace those inputs.

## 4. Obsolete MSc L.T. Jordan / Beyond Borders materials

### Entire event authoring/delivery workflow

DELETE includes the literal `beyond_borders_*` poster/talk Markdown, LaTeX, PDF, PPTX, portable PPTX and final-deliverables ZIP files under `docs/presentation/`; talk-equation assets; logo, photo, slide and poster render trees `Images/general_audience/` and `Images/latex_poster/`; handoff, review and talking-point notes; temporary revision/export trees listed in Wave 1. These are no longer protected merely because they were generated for an event.

Event-only `docs/presentation/evidence/results_story/` and `talk_feedback_3/` are DELETE. They are illustrative event studies/layout evidence without current mathematical-validation or manuscript consumers. `poster_final_audit/` layout/build/DOI/export wrappers are DELETE; its numerical `math_audit.json` is ARCHIVE. Poster profiling wrappers are DELETE after checking their table copies against preserved original profiles; the original timing study and checkpoint records remain available externally/locally respectively. Timing records are historical score-sampler comparisons, not current selected-MLE benchmarks.

`reporting/presentation/` builders, photo/render/asset/slide/poster scripts and optional presentation dependency list are DELETE in Wave 2. Co-retire the relevant event-only configuration/job paths rather than leaving launcher fragments. No active selected-MLE tests, validation drivers, model kernels or reference generators were found importing the presentation package. However, there are numerical exceptions:

| Exact exception | Classification and prerequisite |
|---|---|
| `reporting/presentation/run_long_chain_study.py` | ARCHIVE original score/comparator chain reproduction source |
| `reporting/presentation/analyze_long_chain_study.py` | ARCHIVE original analysis/metric definitions |
| `reporting/presentation/verify_long_chain_study.py` | ARCHIVE saved-number checks |
| `reporting/presentation/prepare_feedback_evidence.py` | ARCHIVE early raw-weighted/reference study source; equivalent modern identity implementations exist |
| `reporting/presentation/audit_final_poster_math.py` | REVIEW until independent estimator-Jacobian finite-difference and exact Gaussian-mixture CDF checks are extracted/preserved as focused tests |
| `tmp/presentation_long_chains/` | ARCHIVE original eight full latent/parameter chains, including historical timing/seed evidence |
| `docs/presentation/evidence/long_chain_study/` | ARCHIVE protocol, original KDE/reference bank, metrics, source and summary arrays |
| `tmp/presentation_feedback_2/diagnostic.npz`, `metadata.json` | ARCHIVE old score-versus-MLE joint diagnostic referenced in target audit |

The auditor's branch/Hastings checks overlap current focused tests, but its selected-estimator derivative and Gaussian-mixture-CDF calculation are useful independent assertions. Extract only these reusable checks in a separately authorized change before retiring the auditor. Its old reference bank is not promoted to current selected-MLE ground truth.

### Manuscript exceptions: do not delete the whole presentation tree

`docs/paper/paper.tex` currently includes three literal files under `docs/presentation/Images/final/`:

```
student_tail_class_definition.png
information_loss_selected_bars.png
privacy_leakage_probability_shift.png
```

All three are KEEP. Privacy is byte-identical to its release-information audit output. The tail-class image is retained as a manuscript illustration even without an identified exact generator. The information-loss figure uses the MAD-ratio generator in `reporting/diagnostics/prepare_meeting_evidence_figures.py`; its input is the **base** `results/release_information_runs/` study, despite a summary filename ending `_100`. Preserve `results/meeting_pack/information_loss_selected_summary_100.csv` and the original paired draws for its five regimes: Normal n=20, Logistic n=20, Student k=1 n=10, Student k=2 n=20, Student k=3 n=20. This is 300 draw files, not a reason to retain every event export.

The full-data/release-analysis drivers and meeting-figure generator remain KEEP until manuscript generation is separated from meeting packaging. Moving those three figures or rewriting LaTeX paths is outside this audit. A later extraction can remove the final event-directory dependencies entirely.

## 5. Historical experiments: what can be discarded versus archived

### Large derived posterior exports: DELETE 4,380 files

The two release-information studies contain verbose CSV files called `full_data_chain_samples.csv` and `mle_only_chain_samples.csv`. Their actual producer is `reporting/diagnostics/run_release_information_full_data.py`:

- Full-data exports are evenly spaced normal posterior quantiles or deterministic inverse-CDF draws from a one-dimensional numerical grid.
- MLE-only exports are analytic normal quantiles or independent seeded weighted resampling of an original centered-MLE-error bank.

These are not Markov trajectories. Keeping observed datasets, released values, original error banks, run metadata, posterior summaries and the original generator preserves the numerical construction. The DELETE rows list 3,600 draw CSVs in `_100_per_regime/` and 780 in the base study; the 300 manuscript-dependent base files are KEEP. Total logical deletion is 1,492,667,456 bytes. Other compact study inputs are ARCHIVE, and must remain accessible to the retained manuscript consumers while needed.

Before removal, archive and restore-test `mle_error_cache/`, `observed_data.csv`, `run_metadata.json` and summaries for those studies, with original code/versions. Older cache names lack the newer estimator-convention field. Recomputing them with the hardened selector would change the experiment. Preserve old banks as historical, not validated current references. `analyze_release_information.py` consumes the exports; retire/redirect old full-run invocations or reconstruct exports outside the checkout as needed.

A tiny deterministic replay used the saved observation row, released `mu_star`, original NPZ `z`, metadata seed/prior/grid settings and AST-extracted pure posterior functions, without calling a sampler or selector. For each of five `dataset000` cases in `_100_per_regime/`, 2,000 values per export were reconstructed:

| Case | Max full-data error | Max MLE-only error | MLE-only bit-identical |
| --- | --- | --- | --- |
| normal_known_var_n10 | 2.22e-16 | 2.22e-16 | False |
| logistic_n10 | 4.44e-16 | 0 | True |
| laplace_n11 | 2.22e-16 | 0 | True |
| student_t_k1_n10 | 2.22e-16 | 0 | True |
| student_t_k2_n10 | 3.75e-16 | 0 | True |

The largest error was 4.44e-16. Four resampled MLE exports were bit-identical as parsed floats; grid/normal values differed by roundoff. This confirms the mechanism, not universal bit-identical regeneration of every file. The initial replay helper failed because an AST future-import node lacked line numbers; applying `ast.fix_missing_locations` fixed the scratch helper. No production code was changed. The final replay used `PYTHONDONTWRITEBYTECODE=1`, avoiding repository caches. Repeat a small gate across all retained regime/environment combinations before the actual Wave 2 removal if exact legacy metric reconstruction is required.

A reproducible replay recipe is: extract `weighted_quantile`, `log_likelihood_grid`, `cdf_from_density`, `grid_summary`, `full_data_posterior`, `mle_only_posterior` from their existing sources; supply NumPy/SciPy/math, QUANTILES/Q_NAMES and metadata as an argparse-like namespace; load saved x-columns and mu_star with roundtrip float parsing; call the functions with the original `z` bank and compare to CSV `mu`. Do not call `centered_mle_errors`, whose current version may generate a different bank.

### Original Markov chains and scientific failures: ARCHIVE, not blanket deletion

`results/student_k1_n50_geometry_runs/` contains roughly 3.79 GB of original long score-Gibbs chains and branch/latent evidence. `results/final_production_v1/` contains roughly 1.79 GB of original score/comparator runs, configurations and parameter/thinned latent diagnostics. Full x-chains are not saved for every production run; this limitation is explicit in metadata. Canonical reports capture conclusions, but summaries cannot support new branch occupancy, tail-residence or source-version analyses from original retained arrays. External compressed archival is justified; they need not occupy the working tree.

The n=50 geometry audit, old broad transition/branch studies, old long-chain presentation study, initial reference-validation and failed first-block runs also retain unique data. Selective external archival must include their generator version, configurations, seeds, summaries and failure records. Some legacy outputs are mismatched-target score comparisons and are useful only with that label.

`results/latest/` is an actual set of directories, not a symlink. It cannot be assumed disposable just from its name. Its historical profiling/audit consumers were identified; archive those outcomes rather than pretending they duplicate every retained run.

### Other permanent deletions and retired workflows

- `results/dashboard_cache/` (238,234,737 B): derived view/density/ranking cache. The Streamlit workflow is obsolete and no active selected-MLE validation consumes it. Some old external reference CSVs are absent, so exact cache regeneration is not assured; this does not justify retaining obsolete UI output.
- `dashboard/`, `scripts/check_dashboard_data.py`, `grace_postprocess_dashboard.sbatch`, `grace_submit_all.sh`: DELETE together in Wave 2. Archive/retire `scripts/run_analysis_pipeline.py` first; it invokes the dashboard preflight. The submit-all wrapper otherwise submits reference/cost/postprocess jobs. Confirm no live jobs call these paths.
- Smoke, redundant legacy acceptance-tuning/trace illustrations and short profile output groups are DELETE as listed in the CSV, with original relevant final data/configuration retained. Do not promote an old smoke study to independent selected-MLE validation.
- Temporary vendored trees `tmp/presentation_diagnostic_deps/` (31,636,738 B) and `tmp/selected_mle_validation_deps/` (73,141,128 B) are DELETE after capturing dependency/replay information and retiring path consumers. Distribution metadata identifies ArviZ 0.22.0, h5netcdf 1.8.1, xarray 2026.7.0, xarray-einstats 0.11.0, mpmath 1.3.0; SymPy is also present. The documented active oracle environment is external (`/tmp/student-mle-oracle-deps-20261003`, SymPy 1.14.0/mpmath 1.3.0), not the in-repo vendor root. Do not rely on a temporary external directory as durable environment recovery.
- Interpreter/JIT/pytest/font caches outside `.venv/` are DELETE Wave 1. Keep `.venv/` and `.git/` entirely for now. Cache removal changes cold-compilation time; no after-cleanup timing comparison to warmed historical timings is valid without accounting for this.
- Six `tmp/selector_hardening_*.txt` logs match retained canonical hardening logs by content hash and are DELETE. Original failed-run logs remain in the evidence package.

## 6. Large recovery archive: verified duplication and safe replacement

The 11-file checkpoint occupies 9,051,650,359 logical bytes, dominated by `worktree.tar` (8,708,372,480 B) and `history.bundle` (338,040,830 B). The tar indexes 17,300 original regular-file members totaling 8,677,297,559 B. All corresponding current paths exist. Fresh current-content SHA-256 comparisons against the historical manifest found:

- 17,297 equal hashes, totaling 8,677,282,812 B.
- Three differing old versions, totaling 14,747 B: README, original location selector and requirements.
- No missing members or comparison errors.

The comparison hashed **current files against the archived manifest**. It did not freshly hash every byte of the tar payload. The tar was indexed and the three differing payloads read directly and checked; September restore checks remain historical evidence, not a new complete tar restore.

| Differing original member | Bytes | Verified archived SHA-256 |
| --- | --- | --- |
| `README.md` | 1536 | `69beb34c58dc0a2520a3d00451259ed39a8f80eccace08396ffb1ae9fab34eeb` |
| `models/student_location_mle.py` | 13067 | `577cac596f05aa30a39e503db65f36b0efffb652ff29b5185c6004617c436b20` |
| `requirements.txt` | 144 | `98b18a2ca24cd134415c52071384f6ac159dc5ffae4e20641f89812d1caac1ed` |

The old selector exactly matches `docs/research/evidence/selected_mle_selector_2026-10-10/student_location_mle_before.py`. Preserve the old README and requirements as small standalone historical records; they are not both already present as identical loose current files. Git bundle refs include historical branches, remotes, stash and Codex refs, which a worktree tar does not capture. Git remote names are not evidence of a complete remote backup.

The checkpoint predates the October proofs, selectors, samplers and fresh banks. **No current verified external recovery snapshot was identified.** It must remain REVIEW until a new selective recovery package is restored successfully. Do not merely copy the giant tar into another nested repository backup.

### Selective external recovery design

1. Use durable storage outside this checkout, preferably a second filesystem/offsite copy. Destination selection and capacity remain owner decisions; no package was created here.
2. Include KEEP research sources/documents/configs/tests and raw current evidence, plus approved ARCHIVE original studies and their source/configuration packages. Exclude DELETE units, rebuildable vendor/JIT/font caches, the old tar and the working `.venv` payload; capture exact environment versions, Python/platform/thread settings and installation constraints instead. Keep the current live environment until replacement is demonstrated.
3. Save Git refs/bundle, current HEAD, status, staged and unstaged binary patches, untracked sources, modes and the index state needed to recover staged distinctions. A `git bundle --all` alone is insufficient for uncommitted or ignored work. Carry relevant old bundle refs/patches if not represented in the replacement. Do not rewrite current refs to build this package.
4. Include the three differing old files above, compact checkpoint manifests/checksums and a mapping from old member paths to current KEEP/ARCHIVE/DELETE outcomes. It is intentional that obsolete event outputs are not recoverable from the replacement; archive verification must test the research retention policy, not restoration of garbage.
5. Compute content hashes and expanded member/mode inventories, verify compressed streams, then restore into an isolated external directory. Check Git refs, staged/unstaged/untracked source states, modes, source hashes, original scientific arrays, manifest resolution and representative small tests without writing into the live checkout.
6. Only after restore approval may Wave 3 discard the tar and remove the externally protected bundle/patch copies locally. Keep compact checkpoint provenance. Metadata referring to the retired tar should later receive an explicit replacement mapping rather than being silently invalidated.

This replacement can avoid duplicated working data and over 9 GB of nested recovery material. The logical ARCHIVE payload is about 6.11 GiB before compression, dominated by original chains, rather than another full 19-GiB tree. Actual compressed size and current Git-bundle size are unmeasured; do not promise a particular compression ratio. Current scientific KEEP files must also be protected in the external snapshot.

## 7. Three removal waves

| Wave | Units | Regular files | Logical bytes | Allocated GiB | Composition |
| --- | --- | --- | --- | --- | --- |
| 1 | 81 | 1278 | 630,807,240 | 0.590 | DELETE 81 |
| 2 | 229 | 20159 | 8,070,814,700 | 7.571 | DELETE 91, ARCHIVE 137, REVIEW 1 |
| 3 | 4 | 4 | 9,046,622,922 | 8.425 | ARCHIVE 3, REVIEW 1 |

### Wave 1 — approve these exact low-risk units first

Selection is **`category == DELETE` and `wave == 1`**, 81 units, 1,278 regular files; 630,807,240 logical bytes and 633,901,056 allocated bytes (0.590 GiB). No ARCHIVE/REVIEW unit is included. Exact scopes:

- `U0006` — `.pytest_cache/`
- `U0009` — `__pycache__/`
- `U0032` — `diagnostics/__pycache__/`
- `U0088` — `docs/presentation/GPT6_ASTRA_PRESENTATION_POSTER_HANDOFF.md`
- `U0089` — `docs/presentation/Images/creodocs_logo.pdf`
- `U0117` — `docs/presentation/Images/general_audience/`
- `U0118` — `docs/presentation/Images/latex_poster/`
- `U0119` — `docs/presentation/beyond_borders_final_deliverables_2026-09-11.zip`
- `U0120` — `docs/presentation/beyond_borders_general_audience.md`
- `U0121` — `docs/presentation/beyond_borders_general_audience.pdf`
- `U0122` — `docs/presentation/beyond_borders_general_audience.pptx`
- `U0123` — `docs/presentation/beyond_borders_general_audience_handoff.md`
- `U0124` — `docs/presentation/beyond_borders_general_audience_portable.pptx`
- `U0125` — `docs/presentation/beyond_borders_research_poster.md`
- `U0126` — `docs/presentation/beyond_borders_research_poster.pdf`
- `U0127` — `docs/presentation/beyond_borders_research_poster.pptx`
- `U0128` — `docs/presentation/beyond_borders_research_poster.tex`
- `U0131` — `docs/presentation/evidence/poster_final_audit/README.md`
- `U0132` — `docs/presentation/evidence/poster_final_audit/build_validation.json`
- `U0133` — `docs/presentation/evidence/poster_final_audit/crossref_andersen.json`
- `U0134` — `docs/presentation/evidence/poster_final_audit/crossref_diagnostics.json`
- `U0135` — `docs/presentation/evidence/poster_final_audit/crossref_hastings.json`
- `U0136` — `docs/presentation/evidence/poster_final_audit/crossref_manifold.json`
- `U0137` — `docs/presentation/evidence/poster_final_audit/figure_provenance.json`
- `U0138` — `docs/presentation/evidence/poster_final_audit/layout_audit.json`
- `U0140` — `docs/presentation/evidence/poster_final_audit/published_sha256.json`
- `U0141` — `docs/presentation/evidence/results_story/`
- `U0142` — `docs/presentation/evidence/talk_feedback_3/`
- `U0146` — `docs/presentation/paris_photo_edit.md`
- `U0147` — `docs/presentation/poster_final_audit.md`
- `U0148` — `docs/presentation/poster_talking_points.md`
- `U0149` — `docs/presentation/slide_review_2026-09-10.md`
- `U0150` — `docs/presentation/submission_review.md`
- `U0151` — `docs/presentation/talk_equations.tex`
- `U0152` — `docs/presentation/talk_feedback_review_2026-09-10.md`
- `U0287` — `docs/research/evidence/selected_mle_selector_2026-10-10/__pycache__/`
- `U0310` — `docs/research/evidence/shifted_student_followup_2026-10-10/__pycache__/`
- `U0416` — `docs/research/evidence/student_n3_fast_path_2026-10-10/__pycache__/`
- `U0687` — `kde_ref/__pycache__/`
- `U0693` — `models/__pycache__/`
- `U0718` — `reporting/__pycache__/`
- `U0722` — `reporting/diagnostics/__pycache__/`
- `U0758` — `reporting/presentation/__pycache__/`
- `U0806` — `results/meeting_pack/.cache/`
- `U0807` — `results/meeting_pack/.mplconfig/`
- `U0889` — `scripts/__pycache__/`
- `U0946` — `tests/__pycache__/`
- `U0978` — `tmp/deck_photo_credit_revision/`
- `U0979` — `tmp/final_deliverables_2026-09-11/`
- `U0980` — `tmp/gibbs-evolution-3d.html`
- `U0982` — `tmp/manifold_x_space.html`
- `U0983` — `tmp/poster_before_findings_refresh/`
- `U0984` — `tmp/poster_before_latex/`
- `U0985` — `tmp/poster_before_technical/`
- `U0986` — `tmp/poster_caption_revision/`
- `U0987` — `tmp/poster_final_review/`
- `U0988` — `tmp/poster_latex/`
- `U0989` — `tmp/presentation_before_long_chains/`
- `U0990` — `tmp/presentation_before_methods_refresh/`
- `U0992` — `tmp/presentation_equation_compatibility/`
- `U0993` — `tmp/presentation_equations_latex/`
- `U0994` — `tmp/presentation_feedback_2/before_feedback_2.pptx`
- `U0997` — `tmp/presentation_feedback_2/plot_summary.json`
- `U0998` — `tmp/presentation_feedback_3/`
- `U0999` — `tmp/presentation_long_chain_review/`
- `U1001` — `tmp/presentation_photo_brightness/`
- `U1002` — `tmp/presentation_polish/`
- `U1003` — `tmp/presentation_results_story/`
- `U1004` — `tmp/presentation_visual_review_2/`
- `U1007` — `tmp/rattle-crossing-3d.html`
- `U1009` — `tmp/remove_deck_photo_credits.py`
- `U1010` — `tmp/remove_poster_permission_caption.py`
- `U1013` — `tmp/selector_hardening_pytest_final.txt`
- `U1014` — `tmp/selector_hardening_pytest_initial.txt`
- `U1015` — `tmp/selector_hardening_pytest_limit.txt`
- `U1016` — `tmp/selector_hardening_pytest_second.txt`
- `U1017` — `tmp/selector_hardening_validation_final.txt`
- `U1018` — `tmp/selector_hardening_validation_initial.txt`
- `U1020` — `tmp/student-mle-manifolds-3d.html`
- `U1021` — `tmp/student-mle-manifolds.html`
- `U1022` — `tmp/student-poor-start-sampler-paths.html`

Checks: quiesce builds/writers; compare each current expanded member set to the approved row fingerprint; verify the three manuscript figures and scientific carve-outs remain excluded; recheck canonical hardening-log equality. Remove only these units after review, not their parents. This wave intentionally retires event notes/exports and rebuildable caches; it does not require preserving the obsolete exports indefinitely. Record the executed expanded path list externally. Caches regenerate; event originals are recoverable from the old checkpoint until Wave 3, but no promise of later event restoration is made.

Completion: measured size reduction agrees reasonably with allocated estimates; unchanged source/model/selector/ref-bank hashes; manuscript image paths still resolve; no active research evidence is missing. Historical event documentation may have intentional retired links; track those instead of resurrecting exports.

### Wave 2 — historical workflow retirement and externalization

The exact scope is all CSV rows with `wave == 2`, with their classifications and prerequisites applied individually. Important units, by size:

| ID | Category | Path | Logical bytes |
| --- | --- | --- | --- |
| U0870 | ARCHIVE | `results/student_k1_n50_geometry_runs/` | 3,787,676,030 |
| U0785 | ARCHIVE | `results/final_production_v1/` | 1,792,289,541 |
| U0851 | DELETE | `results/release_information_runs_100_per_regime/` | 265,986,830 |
| U0853 | DELETE | `results/release_information_runs_100_per_regime/` | 251,542,563 |
| U0852 | DELETE | `results/release_information_runs_100_per_regime/` | 251,406,495 |
| U0850 | DELETE | `results/release_information_runs_100_per_regime/` | 247,610,405 |
| U0783 | DELETE | `results/dashboard_cache/` | 238,234,737 |
| U0849 | DELETE | `results/release_information_runs_100_per_regime/` | 232,766,779 |
| U0836 | DELETE | `results/release_information_runs/` | 185,803,750 |
| U1000 | ARCHIVE | `tmp/presentation_long_chains/` | 113,180,868 |
| U0869 | ARCHIVE | `results/student_k1_n50_geometry_audit/` | 101,688,279 |
| U1012 | DELETE | `tmp/selected_mle_validation_deps/` | 73,141,128 |
| U0796 | ARCHIVE | `results/latest/` | 55,482,796 |
| U0879 | ARCHIVE | `results/student_mle_transition_grid_remaining_k23/` | 51,290,386 |
| U0792 | ARCHIVE | `results/geometry_audit/` | 49,145,863 |
| U0854 | DELETE | `results/release_information_runs_100_per_regime/` | 36,002,072 |
| U0991 | DELETE | `tmp/presentation_diagnostic_deps/` | 31,636,738 |
| U0875 | ARCHIVE | `results/student_mle_transition_grid_k123_n3_10_20_50/` | 31,564,307 |
| U0130 | ARCHIVE | `docs/presentation/evidence/long_chain_study/` | 28,747,491 |
| U0876 | ARCHIVE | `results/student_mle_transition_grid_k1_refined/` | 28,561,210 |
| U0789 | ARCHIVE | `results/final_production_v1_geometry_audit/` | 26,234,655 |
| U0828 | ARCHIVE | `results/refreshed_analysis/` | 25,130,863 |
| U1011 | ARCHIVE | `tmp/selected_mle_reference_validation_20261003/` | 24,898,486 |
| U0837 | DELETE | `results/release_information_runs/` | 21,548,562 |

Permanent deletion here includes the two explicit-member pseudo-chain units, UI/cache/launchers, old derived smoke/trace/tuning outputs, vendor trees and event builders. Externalize original score/comparator production/n50/transition experiments and original historical reproduction sources. Do not archive a configuration still required by a kept test or entrypoint without supplying its reviewed compatibility/input arrangement first. `configs/analysis_run_registry.yaml` remains physically available until its active readers are retired/redirected, even though its intended long-term category is historical ARCHIVE. The same applies to retained manuscript consumers of externally proposed compact release-study inputs.

Prerequisites: verified external member/content manifests and restore; preserve configuration and original source versions alongside each study; redirect or explicitly retire dynamic runset/config/HPC/report consumers; extract the poster auditor's two scientific checks if retiring it; preserve manuscript input exceptions and numerically verify their reconstruction before later thinning; do not replace old bank identities with fresh selector results. Owner review is still required for any scientific retention-policy relaxation beyond this manifest.

Rollback: rehydrate ARCHIVE paths from verified external storage; regenerable draw exports from the original inputs; use recorded source snapshots/patches to restore retired entrypoints. Unique raw chains cannot be rolled back from compact summaries alone. REVIEW auditor is excluded unless separately resolved. Completion: active unit/integration checks pass in a fresh output location, report/manuscript/evidence paths resolve or carry explicit external mappings, old entrypoints are deliberately retired rather than silently broken, and no current failed-result evidence is lost.

### Wave 3 — discard old nested recovery after replacement restore

Exact four units:

- `U0015` — `_archive_repo_cleanup_20260920_checkpoint/history.bundle`
- `U0017` — `_archive_repo_cleanup_20260920_checkpoint/staged.patch`
- `U0019` — `_archive_repo_cleanup_20260920_checkpoint/unstaged.patch`
- `U0020` — `_archive_repo_cleanup_20260920_checkpoint/worktree.tar`

The tar is REVIEW, not approved DELETE. Bundle and staged/unstaged patches are ARCHIVE, with verified replacement/ref coverage required. Potential recovery is 9,046,622,922 logical bytes, 9,046,695,936 allocated bytes (8.425 GiB); seven compact metadata files remain KEEP. This is the largest storage opportunity, but it is **not** Wave 1.

Completion: isolated recovery verifies current October source and ignored scientific arrays, the three differing historical versions, modes and uncommitted/staged distinctions; external checksum/read tests succeed; the replacement mapping is reviewed. Only then discard the old local payload. Recovery thereafter comes from the selective external package, not from Git alone. Do not run repository-preservation scripts with defaults that nest the old backup again.

## 8. Expected final footprint and unresolved decisions

DELETE + ARCHIVE local removal plus the conditionally superseded old tar would release approximately 16.586 GiB of accounted blocks. After all prerequisites, the retained working tree would be about 2.7 GiB including the current environment and Git history, plus directory/new-audit overhead. Research-only retained logical material excluding `.venv`/`.git` is about 0.189 GiB. This is a projection, not a completed cleanup or an estimate of Git compression/GC savings. No environment/Git thinning is proposed in these waves.

Decisions still requiring review:

1. Approve Wave 1's exact 81-row scope; do not infer approval for the recovery tar or raw arrays.
2. Choose durable external historical/recovery storage and retention responsibility. Current external `/tmp` environments and same-filesystem snapshots are not sufficient disaster recovery.
3. Decide whether to keep original legacy n=50/production joint trajectories externally. This audit recommends yes: summaries would lose branch/tail/time-series evidence. Further permanent thinning is a separate scientific retention decision.
4. Authorize a small manuscript/generator extraction later. It would remove presentation-directory dependencies and permit eventual deletion of the 300 retained draw CSVs after MAD reproduction. No changes made here.
5. Resolve independent numerical auditor checks before retiring its 6,194-byte source; do not delay low-risk asset cleanup for this small helper.
6. Treat empty `.aws/`, `.agents/`, `.codex/` as managed REVIEW directories, not clutter. `.git` and `.venv` are KEEP.

Naming/statistical concerns from the blueprint remain: legacy Student/Logistic score methods are not selected-MLE references, unknown cache provenance is not repaired by moving files, and historical “production” labels do not establish current selected-MLE correctness. Cleanup must not erase the documented RATTLE/affine mixing failures or conflate the old selector convention with the hardened one.

## 9. Verification record and smallest next action

Completed read-only checks: full filesystem inventory; Git status and protected hashes; AST/import and exact path/reference searches; Jevgrep discovery limited to relevant code (no massive results indexing); dynamic runpy/spec-from-file loaders and runset/config/HPC paths; manuscript/figure input traces; old archive manifest versus current-content hashes; tar index and three original differing members; three fresh bank hashes; six duplicate-log hashes; five tiny deterministic draw-export replays. No MCMC, pytest suite, large simulation, new backup or deletion was run. Historical tests remain attributed to their reports. The successful replay checks numerical reconstruction, not a fresh sampler validation.

The end-of-audit preservation check passed: all 888 protected files match their initial hashes, HEAD and every original tracked status entry are unchanged, and exactly the two requested audit files were added. The CSV parses with the standard Python field limit (largest field 78,951 characters); all 44,674 non-directory paths are covered once, with matching scope hashes and logical sizes. Ignored scientific arrays were read, not rewritten; only the three fresh bank hashes and original archive-member matches are independently content-verified here, rather than claiming a full new hash manifest of every October ignored array.

**Recommended next action:** execute only the 81 DELETE/Wave-1 units after scope review, with expanded-path verification and manuscript/scientific carve-outs checked. Then stop and measure space. Do not immediately execute Wave 2/3, reorganize models, change algorithms or build a giant backup.
