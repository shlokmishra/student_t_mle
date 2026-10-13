# Collaborator review checkpoint — 12 October 2026

Start with the [review note](../../location_only_meeting_brief.md), then the
[performance report](../../location_only_performance_comparison.md) and
[full trace gallery](../location_chain_diagnostics_2026-10-12/index.html).
The note links the target derivation, sampler corrections and geometry proofs.

This checkpoint changes documentation and adds a figure builder. It changes no
sampler, selector, reference calculation, numerical setting or stored chain.
Tests and the two bounded examples were rerun once. No research experiment was
repeated. The existing LaTeX paper only gains a visible status banner.

## Files

- `checkpoint.json`, `protected_before.json`: pre-edit state and SHA256 hashes
  for 1,824 existing files, excluding environment/compiled caches.
- `documents_before.tar.gz`: only the six documents edited in this task,
  preserving their pre-existing uncommitted contents. This is not a repository
  or data backup.
- `git_status_before.txt`: the already dirty worktree, recorded before edits.
- `figures/`: two new compact views of the existing Student n=10 chains, each
  in PNG/PDF. `provenance.json` gives 55 input hashes, figure hashes and the
  exact annotations. Both rendered PNGs were visually reviewed.
- `validation/`: 403 passing tests, two passing examples and three passing CLI
  checks. `validation.json` records commands, environment, ceilings and exit codes.
  Pytest reports 637.11 seconds; its enclosing process took 665.66 seconds.
- `handoff_inventory.csv`: file-level inventory of the preserved research and
  current additions, with current byte sizes, hashes, Git state and reading tier.
- `verification.json`: preservation, local link and figure-input checks.
- `artifact_manifest.json`: hashes of this checkpoint and the edited documents.

## Sharing and provenance

The inventory's tiers set **reading priority, not permission to omit files**.
`source_and_setup` covers working source, tests, scripts and environment
declarations. `primary_review` covers this note and the latest performance/trace
evidence; `research_support` covers linked theory and earlier validation;
`historical_support` covers earlier retained inputs, manuscript resources and
recovery/provenance records. Some tests load historical source snapshots, so
they must accompany a reproducible test checkout.

For a complete scientific handoff, include every inventory entry plus this
directory's inventory, verification and artifact manifest. Those three files are
excluded from the CSV to avoid recursive hashes. Do not rely on `git archive`:
canonical source is currently untracked, and raw chains/reference banks are
ignored. This task creates no distribution archive and sends nothing externally.
Installed dependencies and compiled caches are deliberately outside the manifest.

The old study manifests describe their original snapshots. They remain unchanged
even where a guide has now been updated. `documents_before.tar.gz` and the
authorized-change list explain those differences. Do not rewrite historical
hashes to describe current documentation.

Regenerate just the two compact figures from the repository root, with a fresh
destination:

```bash
.venv/bin/python scripts/build_location_review_figures.py --out tmp/review-figures-new
```

This reads original chains and the expanded diagnostic audit, checks their
hashes, uses the original burn-in, and plots every stored iteration. PDF metadata
may differ on regeneration; the relevant reproducibility checks are the input
hashes, retained arrays, annotations, axes and plotted data. No byte-identical
image guarantee across rendering environments is claimed.
