"""Reusable components for the mouse-behavior analysis project.

The package contains importable analysis modules.  Command-line entry points
live in the repository's :mod:`scripts` directory so that importing a module
does not accidentally start a video job or parse command-line arguments.
"""

from .version import __version__

__all__ = ["__version__"]
