"""Command-line interface for MCAP to LeRobot conversion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import (
    DEFAULT_ROBOT_TYPE,
    DEFAULT_TASK,
    DIALECT_LEROBOT_ROUNDTRIP,
    DIALECT_XTAC,
    DIALECTS,
    ConverterConfig,
)
from .converter import McapToLeRobotConverter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcap2lerobot",
        description="Convert MCAP episode files into a LeRobot v3 dataset",
    )
    parser.add_argument("inputs", nargs="+", help="MCAP files or directories")
    parser.add_argument("-o", "--output", required=True, help="New dataset directory")
    parser.add_argument("--repo-id", default=None, help="LeRobot repository ID")
    parser.add_argument(
        "--dialect",
        choices=DIALECTS,
        default=DIALECT_XTAC,
        help=(
            f"Source MCAP layout. {DIALECT_XTAC!r} (default): XTacData recordings. "
            f"{DIALECT_LEROBOT_ROUNDTRIP!r}: the ROS2 CDR layout lerobot2mcap emits."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override the fps read from the source (else the source's own metadata)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Task text for every frame. Default: the source's own task "
            f"(the embedded tasks for {DIALECT_LEROBOT_ROUNDTRIP!r}, the task_id for "
            f"{DIALECT_XTAC!r}), else {DEFAULT_TASK!r}"
        ),
    )
    parser.add_argument(
        "--robot-type",
        default=None,
        help=(
            "robot_type for meta/info.json. Default: the source's own value, "
            f"else {DEFAULT_ROBOT_TYPE!r} ({DIALECT_LEROBOT_ROUNDTRIP!r}) or "
            f"'xtac_umi_g1' ({DIALECT_XTAC!r})"
        ),
    )
    roundtrip = parser.add_argument_group(f"{DIALECT_LEROBOT_ROUNDTRIP!r} dialect topic layout")
    roundtrip.add_argument("--state-topic", default="robot/states/data")
    roundtrip.add_argument("--action-topic", default="robot/actions/data")
    roundtrip.add_argument("--state-field", default="observation_state")
    roundtrip.add_argument("--action-field", default="action")
    parser.add_argument(
        "--max-sync-gap-frames",
        type=float,
        default=1.5,
        help="Maximum nearest-neighbor timestamp gap in frame periods",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=0,
        help="Background image staging threads (0 disables; may increase memory use)",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per camera video encoder (default: auto budget across cameras)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    output = Path(args.output).expanduser().resolve()
    config = ConverterConfig(
        dialect=args.dialect,
        state_topic=args.state_topic,
        action_topic=args.action_topic,
        state_field=args.state_field,
        action_field=args.action_field,
        fps=args.fps,
        max_sync_gap_frames=args.max_sync_gap_frames,
        task=args.task,
        robot_type=args.robot_type,
        image_writer_threads=args.image_writer_threads,
        encoder_threads=args.encoder_threads,
    )
    converter = McapToLeRobotConverter(
        inputs=[Path(value) for value in args.inputs],
        output_root=output,
        repo_id=args.repo_id or output.name,
        config=config,
    )
    try:
        result = converter.convert()
    except Exception as error:
        logging.error("Conversion failed: %s", error)
        return 1
    logging.info("LeRobot v3 dataset written to %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
