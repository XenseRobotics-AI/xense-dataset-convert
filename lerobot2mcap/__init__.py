"""LeRobot to MCAP converter."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np

# Fix tabular2mcap bugs
import tabular2mcap.converter.others
import tabular2mcap.mcap_converter
from tabular2mcap.converter.common import ConvertedRow

# Encoder knobs for the all-keyframe video pass. Mutated by main() based on
# CLI args, then read inside _fixed_compressed_video_message_iterator.
VIDEO_ENCODER_PRESET = "ultrafast"
VIDEO_ENCODER_CRF = 23  # libx264 default; visually lossless for robotics


def _fixed_create_foxglove_compressed_image_data(
    frame_timestamp: float, frame_id: str, encoded_data: bytes, format: str
) -> dict:
    """Fixed version that uses 'nanosec' for ROS2 compatibility."""
    return {
        "timestamp": {
            "sec": int(frame_timestamp),
            "nanosec": int((frame_timestamp % 1) * 1_000_000_000),
        },
        "frame_id": frame_id,
        "data": encoded_data,
        "format": format,
    }


def _fixed_compressed_video_message_iterator(
    video_frames: list[np.ndarray],
    fps: float,
    format: str,
    frame_id: str,
    use_foxglove_format: bool = True,
    writer_format: str = "json",
) -> Iterable[ConvertedRow]:
    """
    Fixed version using ffmpeg to encode video with every frame as keyframe.
    Supports H.264 and AV1 formats. Each frame can be decoded independently.
    """
    import os
    import subprocess
    import tempfile

    height, width = video_frames[0].shape[:2]

    # Codec configuration for different formats
    codec_configs = {
        "h264": {
            "encoder": "libx264",
            "ext": "h264",
            "extra_args": [
                "-preset",
                VIDEO_ENCODER_PRESET,
                "-crf",
                str(VIDEO_ENCODER_CRF),
                "-tune",
                "zerolatency",
                "-bf",
                "0",
                "-flags",
                "+global_header",
                "-bsf:v",
                "dump_extra",
            ],
            "frame_marker": 7,  # SPS NAL type
        },
        "av1": {
            "encoder": "libsvtav1",
            "ext": "obu",
            "extra_args": [
                "-svtav1-params",
                "keyint=1:lookahead=0",
            ],
            "frame_marker": 1,  # OBU_SEQUENCE_HEADER
        },
    }

    # Default to h264 if format not supported
    if format not in codec_configs:
        format = "h264"

    config = codec_configs[format]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pipe raw BGR frames straight to ffmpeg — avoids the MJPG round-trip
        # (encode to JPEG, write AVI, ffmpeg re-decode JPEG) that dominated runtime.
        encoded_file = os.path.join(tmpdir, f"encoded.{config['ext']}")
        cmd = (
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",  # cv2/numpy frames are BGR
                "-s",
                f"{width}x{height}",
                "-r",
                str(int(fps)),
                "-i",
                "pipe:0",
                "-c:v",
                config["encoder"],
                "-pix_fmt",
                "yuv420p",
                "-g",
                "1",  # GOP=1, every frame is keyframe
                "-keyint_min",
                "1",
            ]
            + config["extra_args"]
            + [
                "-f",
                config["ext"] if format == "h264" else "ivf",
                "-loglevel",
                "error",
                encoded_file,
            ]
        )
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            for frame in video_frames:
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr_data = proc.stderr.read()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(
                    f"ffmpeg encode failed (code {ret}): "
                    f"{stderr_data.decode(errors='replace')}"
                )
        except BrokenPipeError:
            stderr_data = proc.stderr.read()
            ret = proc.wait()
            raise RuntimeError(
                f"ffmpeg stdin pipe broke (code {ret}): "
                f"{stderr_data.decode(errors='replace')}"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        # Read encoded bitstream
        with open(encoded_file, "rb") as f:
            bitstream = f.read()

        # Split into frames based on format
        if format == "h264":
            frames_data = _split_h264_frames(bitstream)
        else:  # av1
            frames_data = _split_av1_frames(bitstream)

        # Generate messages
        frame_timestamp: float = 0
        frame_timestamp_step = 1 / fps

        for frame_data in frames_data:
            yield ConvertedRow(
                data=_fixed_create_foxglove_compressed_image_data(
                    frame_timestamp=frame_timestamp,
                    frame_id=frame_id,
                    encoded_data=frame_data,
                    format=format,
                ),
                log_time_ns=int(frame_timestamp * 1_000_000_000),
                publish_time_ns=int(frame_timestamp * 1_000_000_000),
            )
            frame_timestamp += frame_timestamp_step


def h264_message_iterator(
    h264_path: Path,
    fps: float,
    frame_id: str,
    format: str = "h264",
) -> Iterable[ConvertedRow]:
    """Yield one ConvertedRow per frame from a pre-encoded all-keyframe .h264 file.

    Used in the v3 fast path: the slice + reencode was already done by ffmpeg in
    ``save_video_slice_h264``, so here we only have to read the bitstream and
    split it on SPS boundaries — no decode, no second encode, no GIL-bound
    Python feeding loop.
    """
    with open(h264_path, "rb") as f:
        bitstream = f.read()
    frames_data = _split_h264_frames(bitstream)

    frame_timestamp: float = 0.0
    frame_timestamp_step = 1.0 / fps
    for frame_data in frames_data:
        yield ConvertedRow(
            data=_fixed_create_foxglove_compressed_image_data(
                frame_timestamp=frame_timestamp,
                frame_id=frame_id,
                encoded_data=frame_data,
                format=format,
            ),
            log_time_ns=int(frame_timestamp * 1_000_000_000),
            publish_time_ns=int(frame_timestamp * 1_000_000_000),
        )
        frame_timestamp += frame_timestamp_step


def _split_h264_frames(bitstream: bytes) -> list[bytes]:
    """Split H.264 Annex B bitstream into frames. Each frame starts with SPS.

    Uses ``bytes.find`` (C-implemented) instead of a per-byte Python scan, so
    splitting a ~10 MB bitstream drops from ~10 s to well under a second.
    """
    frames_data: list[bytes] = []
    n = len(bitstream)

    # Walk all 00 00 01 start-code positions. The 4-byte variant 00 00 00 01
    # is just a 3-byte start code with a leading 00, so this catches both.
    start_positions: list[int] = []
    i = 0
    while True:
        pos = bitstream.find(b"\x00\x00\x01", i)
        if pos < 0:
            break
        start_positions.append(pos)
        i = pos + 3

    # Group NAL units into frames (each frame begins at an SPS, NAL type 7).
    current_frame_start: int | None = None
    for pos in start_positions:
        nal_header = bitstream[pos + 3]  # byte right after 00 00 01
        if (nal_header & 0x1F) != 7:
            continue
        # New SPS — current frame ends here. For the 4-byte variant 00 00 00 01,
        # back up one byte so the leading 00 belongs to this frame's start code.
        frame_start = pos - 1 if pos > 0 and bitstream[pos - 1] == 0 else pos
        if current_frame_start is not None:
            frames_data.append(bitstream[current_frame_start:frame_start])
        current_frame_start = frame_start

    if current_frame_start is not None:
        frames_data.append(bitstream[current_frame_start:n])

    return frames_data


def _split_av1_frames(bitstream: bytes) -> list[bytes]:
    """Split AV1 IVF container into individual frames with sequence header."""
    import struct

    frames_data = []

    # IVF header is 32 bytes
    if len(bitstream) < 32:
        return frames_data

    # Parse IVF header
    signature = bitstream[0:4]
    if signature != b"DKIF":
        # Not IVF format, try raw OBU
        return _split_av1_obu_frames(bitstream)

    # Skip IVF header (32 bytes)
    pos = 32

    # Extract sequence header from first frame (we'll prepend it to all frames)
    seq_header = None

    while pos < len(bitstream):
        if pos + 12 > len(bitstream):
            break

        # IVF frame header: 4 bytes size, 8 bytes timestamp
        frame_size = struct.unpack("<I", bitstream[pos : pos + 4])[0]
        pos += 12  # Skip frame header

        if pos + frame_size > len(bitstream):
            break

        frame_data = bitstream[pos : pos + frame_size]

        # Check if this frame contains sequence header OBU
        if frame_data and (frame_data[0] & 0x78) >> 3 == 1:  # OBU_SEQUENCE_HEADER
            # Extract sequence header for prepending to other frames
            has_size = (frame_data[0] & 0x02) != 0
            if has_size and len(frame_data) > 1:
                # Find OBU size using LEB128
                obu_size = 0
                shift = 0
                idx = 1
                while idx < len(frame_data):
                    byte = frame_data[idx]
                    obu_size |= (byte & 0x7F) << shift
                    idx += 1
                    if (byte & 0x80) == 0:
                        break
                    shift += 7
                seq_header = frame_data[: idx + obu_size]

        # For all-keyframe mode, each frame should be independent
        # Prepend sequence header if not already present
        if seq_header and frame_data and (frame_data[0] & 0x78) >> 3 != 1:
            frame_data = seq_header + frame_data

        frames_data.append(frame_data)
        pos += frame_size

    return frames_data


def _split_av1_obu_frames(bitstream: bytes) -> list[bytes]:
    """Split raw AV1 OBU stream into frames."""
    # For raw OBU, just return the whole bitstream as one frame
    # This is a fallback - ideally we'd parse OBU properly
    return [bitstream] if bitstream else []


# Apply fixes
tabular2mcap.converter.others.create_foxglove_compressed_image_data = (
    _fixed_create_foxglove_compressed_image_data
)
tabular2mcap.converter.others.compressed_video_message_iterator = (
    _fixed_compressed_video_message_iterator
)
tabular2mcap.mcap_converter.compressed_video_message_iterator = (
    _fixed_compressed_video_message_iterator
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .converter import LeRobotConverter

try:
    # The distribution is xense-dataset-convert; this package is one of the four
    # it ships. Looked up by that name rather than this module's, and tolerant of
    # not being installed at all, since the repo is also used as a submodule that
    # a consumer may simply put on sys.path.
    __version__ = version("xense-dataset-convert")
except PackageNotFoundError:
    __version__ = "unknown"

from .logger import get_logger

logger = get_logger("lerobot2mcap")

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_CONVERTER_FUNCTIONS = str(PACKAGE_ROOT / "configs" / "converter_functions.yaml")


def download_dataset(dataset_id: str, episodes: list[int] | None = None) -> Path | None:
    """
    Download a lerobot dataset from Hugging Face Hub.

    Downloads to default HuggingFace cache: ~/.cache/huggingface/lerobot

    Returns:
        Path to the downloaded dataset root, or None if download failed.
    """
    logger.info(f"Downloading: {dataset_id}")
    if episodes:
        logger.info(f"  Episodes: {episodes}")

    try:
        # Use default cache location (don't pass root)
        dataset = LeRobotDataset(dataset_id, episodes=episodes)
        logger.info(
            f"Download complete - Episodes: {dataset.num_episodes}, "
            f"Frames: {dataset.num_frames}, FPS: {dataset.fps}"
        )
        # Return the actual dataset root path
        return Path(dataset.root)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    converter_functions_path: Path,
    chunks: list[int] | None = None,
    episodes: list[int] | None = None,
    intra_workers: int = 8,
    dialect: str = "xtac",
    preserve_source: bool = True,
) -> bool:
    """
    Convert a LeRobot dataset to MCAP format.
    Episodes are processed serially; intra_workers controls how many cameras
    inside one episode are sliced + encoded concurrently.

    Args:
        dataset_root: Root directory of the LeRobot dataset
        output_dir: Output directory for MCAP files
        converter_functions_path: Path to converter_functions.yaml
        episodes: List of episode indices to convert (None = all episodes)
        intra_workers: Per-episode internal parallelism (default: 8)
        dialect: XTac JSON/protobuf (default) or legacy ``lerobot_roundtrip`` CDR.
        preserve_source: Preserve checksummed source payloads for complete XTac return conversion.
    Returns:
        True only when every requested episode was converted; False when any
        episode was skipped or the run failed outright (the cause is logged).
    """
    logger.info(f"Converting dataset: {dataset_root}")
    if episodes:
        logger.info(f"  Episodes: {episodes}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Converter functions: {converter_functions_path}")
    logger.info(f"  Intra-episode workers: {intra_workers}")

    try:
        if dialect == "xtac":
            from xense_dataset_convert._xtac_export import export_xtac
            from .dataset_info import DatasetInfo

            if chunks is not None:
                raise ValueError("XTac export uses episode selection, not chunk selection")
            return export_xtac(
                DatasetInfo(Path(dataset_root) / "meta/info.json"),
                Path(output_dir), episodes, preserve_source, intra_workers,
            )
        if dialect != "lerobot_roundtrip":
            raise ValueError(f"Unknown MCAP dialect: {dialect!r}")
        # Initialize the converter
        converter = LeRobotConverter(
            dataset_root=dataset_root,
            converter_functions_path=converter_functions_path,
            intra_workers=intra_workers,
            dialect=dialect,
            preserve_source=preserve_source,
        )

        # Show conversion plan
        logger.info("\n" + converter.get_conversion_plan(chunks))

        # Perform conversion
        success = converter.convert(
            output_dir=output_dir,
            chunks=chunks,
            episodes=episodes,
        )

        return success

    except Exception as e:
        logger.error(f"Conversion failed: {type(e).__name__}: {e}")
        return False


_LIBX264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)


def _add_video_encoder_args(p: argparse.ArgumentParser) -> None:
    """Attach --video-preset and --video-crf to a subparser.

    These tune the per-frame all-keyframe re-encode. Defaults are chosen for a
    good size/quality trade-off on robotics video (see tests/bench_encoder.sh).
    """
    p.add_argument(
        "--dialect", choices=("xtac", "lerobot_roundtrip"), default="xtac",
        help="MCAP layout (default: xtac). lerobot_roundtrip keeps the legacy CDR exporter.",
    )
    p.add_argument(
        "--preserve-source", action=argparse.BooleanOptionalAction, default=True,
        help="Preserve checksummed source files for exact restoration (XTac default: enabled)",
    )
    p.add_argument(
        "--video-preset",
        choices=_LIBX264_PRESETS,
        default=VIDEO_ENCODER_PRESET,
        help=(
            f"Legacy lerobot_roundtrip libx264 preset "
            f"(default: {VIDEO_ENCODER_PRESET}). Faster presets produce larger files."
        ),
    )
    p.add_argument(
        "--video-crf",
        type=int,
        default=VIDEO_ENCODER_CRF,
        help=(
            f"Legacy lerobot_roundtrip CRF quality (0-51, default: {VIDEO_ENCODER_CRF}). "
            "XTac always uses lossless RGB H.264."
        ),
    )


def main():
    from xense_dataset_convert._progress import configure_cli_logging

    configure_cli_logging()
    parser = argparse.ArgumentParser(
        prog="lerobot2mcap", description="Convert LeRobot datasets to MCAP format"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define download parser arguments
    download_parser = subparsers.add_parser(
        "download", help="Download a LeRobot dataset and convert to MCAP"
    )
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., lerobot/pusht)")
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for MCAP files (default: ./{dataset_name}_mcap)",
    )
    download_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to download (e.g., 0 1 2). If not specified, all episodes will be downloaded.",
    )
    import os

    default_workers = min(8, (os.cpu_count() or 2) // 2)
    default_workers = max(1, default_workers)
    download_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_workers,
        help=(
            f"Per-episode intra-parallelism (default: {default_workers}). "
            f"Episodes themselves run sequentially; within each episode this many "
            f"camera slice+encode jobs run concurrently."
        ),
    )
    _add_video_encoder_args(download_parser)

    # Define
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a LeRobot dataset to MCAP format"
    )
    convert_parser.add_argument(
        "input_dir",
        help="Input directory containing LeRobot dataset (dataset root with meta/info.json)",
    )
    convert_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for MCAP files (default: input_dir/mcap)",
    )
    convert_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to convert (e.g., 0 1 2). If not specified, all episodes will be converted.",
    )
    convert_parser.add_argument(
        "-f",
        "--converter-functions",
        default=DEFAULT_CONVERTER_FUNCTIONS,
        help=f"Path to converter_functions.yaml file (default: {DEFAULT_CONVERTER_FUNCTIONS})",
    )
    convert_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=default_workers,
        help=(
            f"Per-episode intra-parallelism (default: {default_workers}). "
            f"Episodes themselves run sequentially; within each episode this many "
            f"camera slice+encode jobs run concurrently."
        ),
    )
    _add_video_encoder_args(convert_parser)

    args = parser.parse_args()

    # Apply video encoder overrides (mutates module-level config read by the
    # monkey-patched compressed_video iterator).
    preset = getattr(args, "video_preset", None)
    if preset is not None:
        global VIDEO_ENCODER_PRESET
        VIDEO_ENCODER_PRESET = preset
    crf = getattr(args, "video_crf", None)
    if crf is not None:
        global VIDEO_ENCODER_CRF
        VIDEO_ENCODER_CRF = crf

    # Handle download command
    if args.command == "download":
        # Download to default HuggingFace cache (~/.cache/huggingface/lerobot)
        dataset_root = download_dataset(args.dataset_id, args.episodes)
        if dataset_root is None:
            return 1  # Download failed

        logger.info(f"Dataset location: {dataset_root}")

        # MCAP output directory: use -o or default to ./{dataset_name}_mcap
        dataset_name = args.dataset_id.replace("/", "_")
        mcap_output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else Path(f"./{dataset_name}_mcap")
        )
        converter_functions = Path(DEFAULT_CONVERTER_FUNCTIONS)
        chunks = None  # Convert all chunks
        episodes = args.episodes  # Use same episode filter as download
        intra_workers = args.jobs

    elif args.command == "convert":
        # Set parameters from convert command arguments
        dataset_root = Path(args.input_dir).expanduser()
        mcap_output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else dataset_root.resolve().with_name(dataset_root.resolve().name + "_mcap")
        )
        converter_functions = Path(args.converter_functions).expanduser()
        chunks = None  # Convert all chunks
        episodes = args.episodes
        intra_workers = args.jobs

    else:
        # No command provided
        parser.print_help()
        return 0

    # Perform conversion (always happens after download, or standalone)
    if convert_dataset(
        dataset_root,
        mcap_output_dir,
        converter_functions,
        chunks,
        episodes,
        intra_workers,
        dialect=args.dialect,
        preserve_source=args.preserve_source,
    ):
        return 0
    else:
        return 1
