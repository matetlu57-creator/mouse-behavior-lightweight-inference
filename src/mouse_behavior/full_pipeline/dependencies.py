"""Controlled adapters for full-pipeline modules supplied by the runtime.

The historical full pipeline can be inspected and its CLI help can be used
without the private ReID/recovery extensions. A feature that actually needs a
missing extension fails at first use with an actionable error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn


class FullPipelineDependencyError(ImportError):
    """Raised when an external full-pipeline extension is first required."""


def raise_missing_dependency(module_name: str, purpose: str, cause: BaseException) -> NoReturn:
    """Raise a stable diagnostic without hiding import errors inside installed modules."""

    raise FullPipelineDependencyError(
        "The full behavior pipeline requires the external runtime module "
        f"'{module_name}' for {purpose}. Install the version that belongs to "
        "this analysis environment or add its directory to PYTHONPATH."
    ) from cause


class MissingDependencyModule:
    """Module-like object that reports a missing extension on attribute use."""

    def __init__(self, module_name: str, purpose: str, cause: BaseException) -> None:
        self._module_name = module_name
        self._purpose = purpose
        self._cause = cause

    def __getattr__(self, _name: str) -> NoReturn:
        raise_missing_dependency(self._module_name, self._purpose, self._cause)


def missing_dependency_callable(
    module_name: str,
    purpose: str,
    cause: BaseException,
) -> Callable[..., NoReturn]:
    """Return a placeholder for a missing imported class or function."""

    def unavailable(*_args: object, **_kwargs: object) -> NoReturn:
        raise_missing_dependency(module_name, purpose, cause)

    return unavailable
