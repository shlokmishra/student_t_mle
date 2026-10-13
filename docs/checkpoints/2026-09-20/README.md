> **Recovery archive retired on 2026-10-12 at the user's request.** The old
> `_archive_repo_cleanup_20260920_checkpoint/` tar, Git bundle and patches were
> deleted. These documents describe historical checks, not an available restore
> payload. Current source recovery/provenance is described in the
> [cleanup follow-up](../../research/repository_cleanup_followup.md).

# Checkpoint 0: repository preservation and baseline

The first review checkpoint is ready: current state preserved, original
experiment outputs retained, a proposed organization documented, and the
existing numerical baseline established. **113 project tests pass**. The
production samplers and estimator implementations have not been changed.

Preservation verified 17,300 files (8.081 GiB) plus Git history. An isolated
recovery restored all 17,300 files, matched their hashes and rwx modes, and
reproduced both Git patches. Comparing current original non-ignored files to
the checkpoint found only the intentional README edit.

Read in this order:

1. [Inventory and provenance gaps](inventory.md).
2. [Proposed organization and migration contract](organization.md).
3. [Preservation manifest and recovery procedure](preservation.md).
4. [Baseline validation report](baseline.md).
5. [Roadmap](../../roadmap.md) and [evidence ledger](../../evidence_ledger.csv).

The [portable manifest](preservation_manifest.csv) covers original files,
including ignored research artifacts. [Evidence copies](evidence_files.csv)
preserve 13 compact tables/configuration/summary files with original paths and
checksums. The [artifact links](artifact_links.csv) identify critical n=3,
paper and presentation dependencies; the [import map](local_imports.csv) guides
the later code migration. [Preservation summary](preservation-summary.json)
and [recovery drill](restore-drill.json) record archive checks.

Changes made in this checkpoint: preservation/inventory/validation utilities,
the roadmap and ledger, a runnable score-conditioned example, and a corrected
README. One isolated tooling fix restricts default pytest discovery to `tests/`
after the unconfigured command collected third-party tests from `tmp/`.
No historical workflows or research outputs have yet been moved.

The important review decisions are the proposed package boundaries and
compatibility period, which evidence becomes canonical after reconciliation,
and where bulky artifacts will live with an explicit retention policy.
The supplied working agreement calls for review at each checkpoint; deep
refactoring is the next checkpoint. Exact selected-MLE validation and the
location-scale feasibility milestone remain pending, with concrete audit items
recorded in the ledger.
