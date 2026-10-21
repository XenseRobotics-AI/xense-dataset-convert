"""LeRobot dataset -> MCAP, via :mod:`lerobot2mcap`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ConversionError, ConvertResult, Format
from ._support import single_input

ACCEPTED_OPTIONS = frozenset(
    {"converter_functions_path", "chunks", "episodes", "intra_workers", "dialect", "preserve_source"}
)

# Not accepted: video_preset / video_crf. lerobot2mcap reads those from the
# module-level VIDEO_ENCODER_PRESET and VIDEO_ENCODER_CRF globals
# (lerobot2mcap/__init__.py:17) rather than taking them as arguments, so honouring
# them here would mean mutating process-wide state -- wrong under concurrent
# conversions. Set those attributes directly, or use the lerobot2mcap CLI.


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    # Imported here, not at module level: this pulls lerobot and therefore torch.
    import lerobot2mcap

    dataset_root = single_input(inputs, Format.LEROBOT, "the dataset root")
    converter_functions = options.get(
        "converter_functions_path", lerobot2mcap.DEFAULT_CONVERTER_FUNCTIONS
    )

    succeeded = lerobot2mcap.convert_dataset(
        dataset_root=dataset_root,
        output_dir=output,
        converter_functions_path=Path(converter_functions),
        chunks=options.get("chunks"),
        episodes=options.get("episodes"),
        **({"intra_workers": options["intra_workers"]} if "intra_workers" in options else {}),
        **({key: options[key] for key in ("dialect", "preserve_source") if key in options}),
    )
    # convert_dataset catches its own exceptions and reports failure as False;
    # the facade contract is to raise, so the cause is in the log it wrote.
    if not succeeded:
        raise ConversionError(
            f"lerobot2mcap failed to convert {dataset_root} (see the log above)"
        )

    return ConvertResult(
        source=Format.LEROBOT, target=Format.MCAP, output_path=output
    )
