"""Where the version comes from, for installs and for submodule checkouts.

The distribution's metadata is the right answer and the first one tried. It is
absent in a case this repo is deliberately usable in: vendored as a git
submodule, pinned to a commit, never pip-installed. TacFlow-Engine uses it that
way, and puts this string in a provenance field, so "unknown" there means an
ingest event cannot say which converter produced a dataset.

So the fallback reads `pyproject.toml`, which is the single source of truth the
metadata was built from in the first place. It is only reachable from a
checkout -- an installed wheel has no pyproject beside it -- which is exactly
the case the metadata already covers.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DISTRIBUTION = "xense-dataset-convert"
UNKNOWN = "0+unknown"

_VERSION_LINE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def resolve_version() -> str:
    """The distribution version, or the checkout's, or :data:`UNKNOWN`."""
    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:
        pass
    # This file is <repo>/xense_dataset_convert/_version.py, so the checkout
    # root is two levels up.
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return UNKNOWN
    # The first `version = "..."` under [project]; `requires-python` and any
    # dependency pins are not of that shape, so a plain search is enough.
    match = _VERSION_LINE.search(text)
    return match.group(1) if match else UNKNOWN
