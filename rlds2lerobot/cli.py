"""Command-line entry point for RLDS to LeRobot conversion."""

from __future__ import annotations

import argparse

from .config import ConverterConfig
from .converter import RldsToLeRobotConverter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert RLDS/TFDS datasets to LeRobot v3")
    parser.add_argument("input", help="Generated TFDS directory, e.g. droid/1.0.0")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--task")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        help="Background image staging threads (0 disables; may increase memory use)",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        help="Threads per camera video encoder (default: auto budget across cameras)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable episode/frame progress bars")
    args = parser.parse_args(argv)
    config = ConverterConfig.from_yaml(args.config) if args.config else ConverterConfig()
    values = {
        key: value
        for key, value in {
            "fps": args.fps,
            "task": args.task,
            "max_episodes": args.max_episodes,
            "image_writer_threads": args.image_writer_threads,
            "encoder_threads": args.encoder_threads,
        }.items()
        if value is not None
    }
    values["show_progress"] = not args.no_progress
    if values:
        from dataclasses import replace
        config = replace(config, **values)
    RldsToLeRobotConverter(args.input, args.output, config, args.split, args.overwrite).convert()
    return 0
