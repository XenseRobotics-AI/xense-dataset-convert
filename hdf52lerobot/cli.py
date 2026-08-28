"""Command line entry point for ``hdf52lerobot``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PAIR_MODES, QUAT_ORDERS, ConverterConfig
from .converter import Hdf5ToLeRobotConverter
from .logger import get_logger

logger = get_logger("hdf52lerobot.cli")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "hdf52lerobot.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdf52lerobot",
        description="Convert UMI-rig HDF5/MKV recordings into a LeRobot v3.0 dataset.",
    )
    parser.add_argument(
        "--input",
        "-i",
        action="append",
        required=True,
        help="An episode directory, or a directory of episode directories. Repeatable.",
    )
    parser.add_argument("--output", "-o", required=True, help="Dataset root to create.")
    parser.add_argument("--task", help="Task description stored in meta/tasks.parquet.")
    parser.add_argument(
        "--config",
        help=f"Stream-rule config (default: {DEFAULT_CONFIG} when it exists).",
    )
    parser.add_argument("--fps", type=int, help="Declared dataset frame rate.")
    parser.add_argument("--robot-type", dest="robot_type", help="robot_type in meta/info.json.")
    parser.add_argument("--quat-order", dest="quat_order", choices=QUAT_ORDERS)
    parser.add_argument("--pair", choices=PAIR_MODES, help="Cross-stream pairing mode.")
    parser.add_argument(
        "--max-sync-gap-frames",
        dest="max_sync_gap_frames",
        type=float,
        help="Reject the episode when a stream sits further than this from the master clock.",
    )
    parser.add_argument(
        "--warn-sync-gap-frames",
        dest="warn_sync_gap_frames",
        type=float,
        help="Report (but accept) frames further than this from the master clock.",
    )
    parser.add_argument("--crf", type=int, help="Encoder quality (lower is better).")
    parser.add_argument("--gop", type=int, help="Keyframe interval of the output videos.")
    parser.add_argument("--preset", help="libx264 preset.")
    parser.add_argument("--workers", type=int, help="Camera keys encoded in parallel.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete the output directory if it exists."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    config = ConverterConfig.from_yaml(config_path) if config_path else ConverterConfig()
    config = config.with_overrides(
        fps=args.fps,
        robot_type=args.robot_type,
        task=args.task,
        quat_order=args.quat_order,
        pair=args.pair,
        max_sync_gap_frames=args.max_sync_gap_frames,
        warn_sync_gap_frames=args.warn_sync_gap_frames,
        crf=args.crf,
        gop=args.gop,
        preset=args.preset,
        workers=args.workers,
    )

    converter = Hdf5ToLeRobotConverter(
        inputs=args.input,
        output_root=args.output,
        config=config,
        overwrite=args.overwrite,
    )
    try:
        converter.convert()
    except Exception as error:  # surfaced as a clean CLI failure, not a traceback
        logger.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
