"""Models grouped by inferred parameter and distribution family.

Use ``models.loc`` and ``models.loc_scale`` for new code. Historical flat names
remain lazy aliases to the same modules. See ``models/README.md`` for targets.
"""

from importlib import import_module
from ._compat import LEGACY_MODULES, install_aliases

install_aliases()

__all__ = ["loc", "loc_scale", "scale", "shared", "model_registry"]


def __getattr__(name):
    if name in __all__:
        module = import_module("models." + name)
        globals()[name] = module
        return module
    if name in LEGACY_MODULES:
        module = import_module("models." + LEGACY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
