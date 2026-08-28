"""Command-line interface for generic LeRobot to Zarr conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

from .logger import get_logger
from .zarr_converter import LeRobotZarrConverter, load_info_json, open_lerobot_dataset

logger = get_logger("lerobot2zarr.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot2zarr",
        description="Convert a LeRobot dataset to a generic, schema-described Zarr store",
    )
    parser.add_argument(
        "input_dir", help="LeRobot dataset root containing meta/info.json"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output path (default: input_dir/zarr_conversion.zarr)"
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repository ID if different from the input directory name",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Only load the selected episode IDs",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Convert only the first N loaded frames"
    )
    parser.add_argument(
        "--chunk-mib",
        type=float,
        default=8.0,
        help="Target chunk size for non-image arrays in MiB (default: 8)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.input_dir).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else dataset_root / "zarr_conversion.zarr"
    )
    if args.chunk_mib <= 0:
        raise SystemExit("--chunk-mib must be greater than zero")

    try:
        info = load_info_json(dataset_root)
        dataset = open_lerobot_dataset(dataset_root, args.repo_id, args.episodes)
        converter = LeRobotZarrConverter(
            dataset=dataset,
            output_path=output_path,
            feature_metadata=info.get("features", {}),
            source={
                "dataset_root": str(dataset_root),
                "repo_id": args.repo_id,
                "codebase_version": info.get("codebase_version"),
                "fps": info.get("fps"),
                "selected_episodes": args.episodes,
            },
            chunk_bytes=int(args.chunk_mib * 1024 * 1024),
        )
        converter.convert(limit=args.limit)
    except Exception as error:
        logger.error(f"Zarr conversion failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
