"""One entry point for every dataset conversion this repository ships.

    from xense_dataset_convert import convert, available, Format

    convert(Format.UMI_HDF5, Format.LEROBOT,
            inputs="/data/SuHang", output="/data/SuHang_lerobot",
            task="pick up the cup")

The four converter packages are left exactly as they are; this layer only
reconciles their interfaces, which differ in shape (function vs class), in what
they accept (a path vs an already-opened dataset), and in how they report
failure (``False`` vs an exception).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from .registry import available, load_adapter
from .types import ConversionError, ConvertResult, Format, UnsupportedConversion

__all__ = [
    "convert",
    "available",
    "Format",
    "ConvertResult",
    "ConversionError",
    "UnsupportedConversion",
    "__version__",
]

try:
    # Single source of truth is pyproject.toml; a checkout used as a submodule
    # without being installed has no distribution metadata to read.
    __version__ = version("xense-dataset-convert")
except PackageNotFoundError:
    __version__ = "0+unknown"


def convert(
    source: Format,
    target: Format,
    inputs: str | Path | Sequence[str | Path],
    output: str | Path,
    **options: Any,
) -> ConvertResult:
    """Convert a dataset from one format to another.

    Args:
        source: Format being read. Build it from a config value with
            ``Format("umi_hdf5")``.
        target: Format being written.
        inputs: One path, or several. What counts as an input differs by
            source format -- a dataset root for ``LEROBOT``, files or a
            directory of them for ``MCAP``, recording directories for
            ``UMI_HDF5`` -- and each adapter says so when given the wrong shape.
        output: Where to write. Created by the underlying converter.
        **options: Passed to the underlying converter. Unknown names raise
            rather than being ignored, since these usually arrive from a config
            file where a typo would otherwise pass silently.

    Returns:
        A :class:`ConvertResult` naming the output and whatever the adapter
        could report for free.

    Raises:
        UnsupportedConversion: No converter exists for this pair.
        ConversionError: The conversion ran and did not produce a dataset.
        TypeError: ``source``/``target`` are not :class:`Format` members, or an
            option name is not accepted for this pair.
    """
    _require_format(source, "source")
    _require_format(target, "target")

    adapter = load_adapter(source, target)
    _reject_unknown_options(adapter, source, target, options)

    return adapter.convert(_as_paths(inputs), Path(output).expanduser(), options)


def _require_format(value: Any, argument: str) -> None:
    if not isinstance(value, Format):
        raise TypeError(
            f"{argument} must be a Format, got {type(value).__name__}. "
            f"Build one from a config value with Format({value!r})."
        )


def _as_paths(inputs: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        return [Path(inputs).expanduser()]
    paths = [Path(entry).expanduser() for entry in inputs]
    if not paths:
        raise ValueError("inputs is empty")
    return paths


def _reject_unknown_options(
    adapter: Any, source: Format, target: Format, options: dict[str, Any]
) -> None:
    unknown = sorted(set(options) - adapter.ACCEPTED_OPTIONS)
    if unknown:
        accepted = ", ".join(sorted(adapter.ACCEPTED_OPTIONS)) or "(none)"
        raise TypeError(
            f"{source} -> {target} does not accept "
            f"{', '.join(repr(name) for name in unknown)}. Accepted: {accepted}"
        )
