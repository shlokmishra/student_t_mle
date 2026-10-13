"""Strict historical source checks across the 2026-10-12 layout migration.

Old provenance stays immutable. Both pre-move bytes and current bytes must match
the documented migration; a new source hash is never reported as an old hash.
"""

import hashlib
import json
from pathlib import Path
import tarfile


def verify_recorded_source(root, recorded_path, expected_sha256):
    root = Path(root)
    path = root / recorded_path
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256:
        return {"recorded_path": recorded_path, "status": "exact_source_match"}
    checkpoint = root / "docs/checkpoints/2026-10-12-model-layout"
    migration = json.loads((checkpoint / "migration.json").read_text())
    entry = next((r for r in migration["files"] if r["old"] == recorded_path), None)
    if entry is None or entry["before_sha256"] != expected_sha256:
        raise ArithmeticError("recorded source not covered by verified migration: " + recorded_path)
    current = root / entry["new"]
    if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != entry["after_sha256"]:
        raise ArithmeticError("source changed after layout migration: " + entry["new"])
    with tarfile.open(checkpoint / "source_before.tar.gz", "r:gz") as archive:
        original = archive.extractfile(recorded_path).read()
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ArithmeticError("pre-migration source archive mismatch: " + recorded_path)
    return {"recorded_path": recorded_path, "current_path": entry["new"],
            "status": "verified_layout_migration", "current_sha256": entry["after_sha256"]}


def verify_recorded_sources(root, sources, paths=None):
    """Check a recorded hash map, accepting requested old or canonical paths."""
    if paths is None:
        return [verify_recorded_source(root, path, digest) for path, digest in sources.items()]
    checkpoint = Path(root) / "docs/checkpoints/2026-10-12-model-layout/migration.json"
    entries = json.loads(checkpoint.read_text())["files"]
    inverse = {entry["new"]: entry["old"] for entry in entries}
    checks = []
    for path in paths:
        key = path if path in sources else inverse.get(path, path)
        if key not in sources:
            raise ArithmeticError("source missing from recorded provenance: " + path)
        checks.append(verify_recorded_source(root, key, sources[key]))
    return checks


def verify_archived_source(root, recorded_path, expected_sha256):
    """Verify historical bytes only; does NOT certify equivalence to current code.

    Use explicitly when reading old evidence after a scientific algorithm change.
    The normal verifier remains strict and rejects modified current algorithms.
    """
    root = Path(root)
    snapshots = [
        root / 'docs/checkpoints/2026-10-12-model-layout/source_before.tar.gz',
        root / 'docs/research/evidence/location_only_correctness_2026-10-12/source_before.tar.gz',
    ]
    for snapshot in snapshots:
        if not snapshot.is_file():
            continue
        with tarfile.open(snapshot, 'r:gz') as archive:
            try:
                content = archive.extractfile(recorded_path).read()
            except KeyError:
                continue
        if hashlib.sha256(content).hexdigest() == expected_sha256:
            return dict(recorded_path=recorded_path, status='verified_archived_source_only',
                        archive=str(snapshot.relative_to(root)), current_numerical_equivalence=False)
    raise ArithmeticError('no matching archived source: ' + recorded_path)
