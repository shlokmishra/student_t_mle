"""Exact old module names for saved scripts, pickles and historical replays.

One alias table replaces twenty flat forwarding files. No numerical library is
loaded until requested; numerical implementations execute only at their new path.
"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

LEGACY_MODULES = {
    "loc_student": "loc.student.score_gibbs",
    "loc_student_rattle": "loc.student.score_rattle",
    "loc_student_selected_mle": "loc.student.selected_gibbs",
    "loc_student_selected_mle_block": "loc.student.selected_block",
    "loc_student_selected_mle_rattle": "loc.student.selected_rattle",
    "loc_student_selected_mle_independence": "loc.student.independence",
    "student_location_mle": "loc.student.mle",
    "student_n3_endpoint": "loc.student.n3_endpoint",
    "student_mu_mh": "loc.student.mu_mh",
    "loc_affine": "loc.affine",
    "loc_logistic": "loc.logistic.score_gibbs",
    "loc_logistic_rattle": "loc.logistic.score_rattle",
    "logistic_gibbs_numba": "loc.logistic._numba",
    "loc_laplace": "loc.laplace.median_gibbs",
    "loc_laplace_rattle": "loc.laplace.facet_rattle",
    "laplace_gibbs_numba": "loc.laplace._numba",
    "loc_scale_student": "loc_scale.student_gibbs",
    "loc_scale_student_rattle": "loc_scale.student_rattle",
    "loc_scale_laplace": "loc_scale.laplace_gibbs",
    "numba_math": "shared.numba_math",
}


class _AliasLoader(Loader):
    def create_module(self, spec):
        module = import_module("models." + LEGACY_MODULES[spec.name.split(".")[1]])
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Python assigns the alias spec even to an existing module returned by
        # create_module. Restore the real identity for reloads and Numba caches.
        module.__spec__ = self.canonical_spec


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("models.") and fullname[7:] in LEGACY_MODULES:
            return spec_from_loader(fullname, _AliasLoader())
        return None


def install_aliases():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_AliasFinder())
