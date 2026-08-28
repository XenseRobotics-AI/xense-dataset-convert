"""Which adapter serves which pair of formats.

The table stores *module paths as strings*, never imported adapters. Adapters
import their converter package, and ``lerobot2mcap`` imports lerobot (and so
torch) at module level -- resolving the table eagerly would make every caller
pay for that, including one converting HDF5 to LeRobot, which needs neither.
``tests/test_xense_dataset_convert.py`` guards this.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol

from .types import Format, UnsupportedConversion

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path
    from typing import Any

    from .types import ConvertResult


class Adapter(Protocol):
    """The shape every module in ``adapters/`` provides."""

    ACCEPTED_OPTIONS: frozenset[str]

    def convert(
        self,
        inputs: "Path | list[Path]",
        output: "Path",
        options: "dict[str, Any]",
    ) -> "ConvertResult": ...


_ADAPTERS: dict[tuple[Format, Format], str] = {
    (Format.LEROBOT, Format.MCAP): "xense_dataset_convert.adapters.lerobot_to_mcap",
    (Format.LEROBOT, Format.ZARR): "xense_dataset_convert.adapters.lerobot_to_zarr",
    (Format.MCAP, Format.LEROBOT): "xense_dataset_convert.adapters.mcap_to_lerobot",
    (Format.UMI_HDF5, Format.LEROBOT): "xense_dataset_convert.adapters.umi_hdf5_to_lerobot",
}


def available() -> list[tuple[Format, Format]]:
    """Every ``(source, target)`` pair that can be converted.

    Answering this imports nothing, so an engine can enumerate what is on offer
    without loading torch.
    """
    return list(_ADAPTERS)


def load_adapter(source: Format, target: Format) -> Adapter:
    """Import and return the adapter for one pair of formats."""
    try:
        module_path = _ADAPTERS[(source, target)]
    except KeyError:
        raise UnsupportedConversion(
            f"Cannot convert {source} to {target}. Available: "
            + ", ".join(f"{one} -> {other}" for one, other in _ADAPTERS)
        ) from None
    return importlib.import_module(module_path)  # type: ignore[return-value]
