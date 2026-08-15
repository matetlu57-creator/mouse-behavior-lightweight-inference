"""Helpers used by the temporary root-level compatibility modules.

The project historically exposed its modules from the repository root.  The
new source layout keeps those import paths working while the implementation
lives under ``src/mouse_behavior``.  This file is intentionally private and
is not part of the public analysis API.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


_MODULE_METADATA = {
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__file__",
    "__cached__",
    "__builtins__",
}


def reexport(module_name: str, namespace: dict[str, Any]) -> ModuleType:
    """Expose an implementation module through a legacy module namespace.

    Private project helpers are included deliberately: a few existing tests
    and downstream notebooks use them as lightweight diagnostic utilities.
    Python's own module metadata is excluded so traceback paths and direct
    file-based imports continue to describe the compatibility file.
    """

    implementation = import_module(module_name)
    for name, value in vars(implementation).items():
        if name not in _MODULE_METADATA:
            namespace[name] = value
    namespace.setdefault(
        "__all__",
        [name for name in vars(implementation) if name not in _MODULE_METADATA],
    )
    return implementation
