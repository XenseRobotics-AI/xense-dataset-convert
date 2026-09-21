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
        description="Convert XTac, ROS2, DAS-Ego, or DAS-UMI MCAP files into a LeRobot v3 dataset",
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
    roundtrip = parser.add_argument_group(
        f"{DIALECT_LEROBOT_ROUNDTRIP!r} dialect topic layout"
    )
    roundtrip.add_argument("--state-topic", default="robot/states/data")
    roundtrip.add_argument("--action-topic", default="robot/actions/data")
    roundtrip.add_argument("--state-field", default="observation_state")
    roundtrip.add_argument("--action-field", default="action")
    parser.add_argument(
        "--profile",
        choices=("auto", "ros2", "das-ego", "das-umi"),
        default="auto",
        help="Input layout (default: auto-detect ROS2, DAS-Ego, or merged DAS-UMI Protobuf)",
    )
    parser.add_argument(
        "--max-sync-gap-frames",
        type=float,
        default=1.5,
        help="Maximum nearest-neighbor timestamp gap in frame periods",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run DAS-UMI preflight and reports without writing a dataset",
    )
    parser.add_argument(
        "--export-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For DAS-UMI, write next-frame absolute action (default: enabled)",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "DAS-UMI semantic schema. v2 keeps acquisition timestamps in an "
            "audit sidecar and removes duplicate control observations (default: 2)"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="DAS-UMI preflight and conversion report directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="DAS-UMI YAML mapping and quality thresholds",
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
        profile=args.profile,
        config=args.config,
        check_only=args.check_only,
        export_action=args.export_action,
        schema_version=args.schema_version,
        report_dir=args.report_dir,
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
    if args.check_only:
        report_dir = (
            (args.report_dir or output.parent / f"{output.name}_reports")
            .expanduser()
            .resolve()
        )
        logging.info("Preflight passed; reports written to %s", report_dir)
    else:
        logging.info("LeRobot v3 dataset written to %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
