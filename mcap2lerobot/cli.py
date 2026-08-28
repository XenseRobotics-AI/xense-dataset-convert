"""Command-line interface for MCAP to LeRobot conversion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ConverterConfig
from .converter import McapToLeRobotConverter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcap2lerobot",
        description="Convert ROS2 MCAP episode files into a LeRobot v3 dataset",
    )
    parser.add_argument("inputs", nargs="+", help="MCAP files or directories")
    parser.add_argument("-o", "--output", required=True, help="New dataset directory")
    parser.add_argument("--repo-id", default=None, help="LeRobot repository ID")
    parser.add_argument("--fps", type=int, default=None, help="Override inferred FPS")
    parser.add_argument("--task", default="Converted from MCAP")
    parser.add_argument("--robot-type", default="mcap")
    parser.add_argument("--state-topic", default="robot/states/data")
    parser.add_argument("--action-topic", default="robot/actions/data")
    parser.add_argument("--state-field", default="observation_state")
    parser.add_argument("--action-field", default="action")
    parser.add_argument(
        "--max-sync-gap-frames",
        type=float,
        default=1.5,
        help="Maximum nearest-neighbor timestamp gap in frame periods",
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
        state_topic=args.state_topic,
        action_topic=args.action_topic,
        state_field=args.state_field,
        action_field=args.action_field,
        fps=args.fps,
        max_sync_gap_frames=args.max_sync_gap_frames,
        task=args.task,
        robot_type=args.robot_type,
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
