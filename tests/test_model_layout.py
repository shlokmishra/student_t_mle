"""Import identity and strict historical provenance after module relocation."""

import hashlib
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from diagnostics.source_provenance import verify_recorded_source, verify_recorded_sources
from models._compat import LEGACY_MODULES
from models.model_registry import model_validity_rows

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "docs/checkpoints/2026-10-12-model-layout"


@pytest.mark.parametrize("old,new", LEGACY_MODULES.items())
def test_old_import_is_same_module_with_canonical_identity(old, new):
    canonical = importlib.import_module("models." + new)
    historical = importlib.import_module("models." + old)
    assert historical is canonical
    assert historical.__name__ == "models." + new
    assert historical.__spec__.name == "models." + new
    assert not (ROOT / "models" / (old + ".py")).exists()


@pytest.mark.parametrize("old_first", [False, True])
def test_import_order_and_private_compiled_helpers(old_first):
    names = ["models.loc_student", "models.loc.student.score_gibbs"]
    if not old_first:
        names.reverse()
    script = (
        "import importlib; "
        f"a=importlib.import_module({names[0]!r}); b=importlib.import_module({names[1]!r}); "
        "assert a is b; assert a.__spec__.name=='models.loc.student.score_gibbs'; "
        "assert a._initial_x is b._initial_x; "
        "assert a.psi(.2,2.)==b.psi(.2,2.)"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, timeout=60)


def test_package_import_does_not_load_joint_samplers():
    subprocess.run([sys.executable, "-c",
        "import models,sys; assert 'jax' not in sys.modules; "
        "assert 'models.loc_scale.student_gibbs' not in sys.modules"],
        cwd=ROOT, check=True, timeout=60)


def test_moved_selector_and_recorded_hash_resolve():
    entry = next(r for r in json.loads((CHECKPOINT / "migration.json").read_text())["files"]
                 if r["old"] == "models/student_location_mle.py")
    checks = verify_recorded_sources(ROOT, {entry["old"]: entry["before_sha256"]}, [entry["new"]])
    assert checks[0]["status"] == "verified_layout_migration"
    assert hashlib.sha256((ROOT / entry["new"]).read_bytes()).hexdigest() == entry["before_sha256"]


def test_unrecognized_source_hash_cannot_bypass_integrity_checks():
    with pytest.raises(ArithmeticError, match="not covered"):
        verify_recorded_source(ROOT, "models/loc_student.py", "0" * 64)


def test_layout_bodies_and_explicit_scientific_revisions():
    class WithoutImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

    entries = json.loads((CHECKPOINT / "migration.json").read_text())["files"]
    with tarfile.open(CHECKPOINT / "source_before.tar.gz", "r:gz") as archive:
        for entry in entries:
            if not entry["old"].startswith("models/"):
                continue
            before = ast.parse(archive.extractfile(entry["old"]).read())
            current = (ROOT / entry["new"]).read_bytes()
            revision = ROOT / "docs/research/evidence/location_only_correctness_2026-10-12"
            changes = json.loads((revision / "source_changes.json").read_text())
            if entry["new"] in changes:
                row = changes[entry["new"]]
                assert hashlib.sha256(current).hexdigest() == row["after_sha256"]
                assert row["before_sha256"] == entry["after_sha256"]
                with tarfile.open(revision / "source_before.tar.gz", "r:gz") as frozen:
                    current = frozen.extractfile(entry["new"]).read()
                assert hashlib.sha256(current).hexdigest() == row["before_sha256"]
            after = ast.parse(current)
            assert ast.dump(WithoutImports().visit(before)) == ast.dump(WithoutImports().visit(after)), entry["new"]


def test_post_migration_source_changes_are_not_accepted(tmp_path):
    import shutil
    relative = "docs/checkpoints/2026-10-12-model-layout"
    destination = tmp_path / relative
    destination.mkdir(parents=True)
    shutil.copyfile(CHECKPOINT / "migration.json", destination / "migration.json")
    entry = next(r for r in json.loads((CHECKPOINT / "migration.json").read_text())["files"]
                 if r["old"] == "models/loc_student.py")
    changed = tmp_path / entry["new"]
    changed.parent.mkdir(parents=True)
    changed.write_text("# Changed computation\n")
    with pytest.raises(ArithmeticError, match="changed after layout migration"):
        verify_recorded_source(tmp_path, entry["old"], entry["before_sha256"])


def test_frozen_block_replay_still_uses_the_same_dependencies():
    from importlib.util import spec_from_file_location, module_from_spec
    from models.loc.student import selected_rattle, score_gibbs
    original = ROOT / "docs/research/evidence/student_n3_fast_path_2026-10-10/original_block.py"
    spec = spec_from_file_location("models._layout_frozen_block", original)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.selected_state is selected_rattle.selected_state
    assert module.legacy is score_gibbs


def test_registry_does_not_claim_score_matches_selected_mle_reference():
    rows = model_validity_rows()
    for family in ("student_t", "logistic"):
        legacy = [row for row in rows if row["model"] == family and row["method"] in {"gibbs", "rattle"}]
        assert all(row["target_matches_reference"] is False for row in legacy)
        refs = [row for row in rows if row["model"] == family and row["method"] == "raw_weighted_mc"]
        assert refs[0]["target_description"] == "selected_global_location_mle_equals_mu_star"
