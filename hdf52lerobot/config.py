"""Configuration for the UMI-rig HDF5 layout.

Everything that ties the converter to a particular rig lives here: which
filename maps to which semantic stream, and which streams get concatenated
into ``observation.state``. Adding a head tracker or a third camera is a
config change, not a code change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

VIDEO = "video"
POSE = "pose"
GRIPPER = "gripper"
KINDS = (VIDEO, POSE, GRIPPER)

QUAT_ORDERS = ("wxyz", "xyzw")
PAIR_MODES = ("index", "nearest")

# Field suffixes contributed to observation.state by each stream kind.
POSE_SUFFIXES = ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
GRIPPER_SUFFIXES = ("pos",)


@dataclass(frozen=True)
class StreamRule:
    """Map a source filename to a semantic stream.

    Args:
        pattern: Regex matched against the file name (not the full path).
            Named groups are substituted into ``key``; the ``lo``/``hi`` groups
            of a gripper rule are read as its calibration bounds.
        kind: One of ``video``, ``pose``, ``gripper``.
        key: Stream key template, e.g. ``"{side}_wrist"``. For videos this
            becomes the feature ``observation.images.<key>``; for pose/gripper
            it prefixes the state field names.
    """

    pattern: str
    kind: str
    key: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown stream kind {self.kind!r}, expected one of {KINDS}")
        re.compile(self.pattern)

    def match(self, filename: str) -> tuple[str, dict[str, str]] | None:
        """Return ``(resolved_key, named_groups)`` when the rule applies."""
        found = re.match(self.pattern, filename)
        if found is None:
            return None
        groups = {name: value for name, value in found.groupdict().items() if value is not None}
        return self.key.format(**groups), groups


# The bimanual TacCap/UMI handheld rig: head camera, two wrist cameras, four
# tactile sensors, two Vive trackers, two openv2 jaw encoders.
DEFAULT_STREAM_RULES: tuple[StreamRule, ...] = (
    StreamRule(r"^head_.*_mvs_mono_.*\.mkv$", VIDEO, "head"),
    StreamRule(r"^(?P<side>left|right)_umi_.*_mvs_mono_.*\.mkv$", VIDEO, "{side}_wrist"),
    StreamRule(
        r"^(?P<side>left|right)_umi_.*_tactile_(?P<pos>left|right)_.*\.mkv$",
        VIDEO,
        "{side}_tactile_{pos}",
    ),
    StreamRule(r"^(?P<side>left|right)_umi_.*_tracker_.*\.hdf5$", POSE, "{side}_tcp"),
    StreamRule(
        r"^(?P<side>left|right)_umi_.*_openv2_(?P<lo>[\d.]+)_(?P<hi>[\d.]+)\.hdf5$",
        GRIPPER,
        "{side}_gripper",
    ),
)

DEFAULT_STATE_LAYOUT: tuple[str, ...] = (
    "left_tcp",
    "left_gripper",
    "right_tcp",
    "right_gripper",
)


@dataclass
class ConverterConfig:
    """Settings for one conversion run.

    Defaults describe the SuHang UMI recordings and produce a dataset
    structurally identical to ``TacCap`` -- same 20-dim state layout, same v3.0
    codebase version -- but tagged ``bi_rdt_gripper``, because the rig's gripper
    and its planar tactile sensors are different hardware.
    """

    fps: int = 20
    """Declared dataset frame rate. Not inferred: the recorder's per-device
    arrival timestamps jitter enough that a median-based estimate is unstable."""

    robot_type: str = "bi_rdt_gripper"
    task: str = "Converted from UMI HDF5 recording"

    quat_order: str = "xyzw"
    """Component order of the tracker's 4 rotation columns. The UMI rig writes
    ``[x, y, z, qx, qy, qz, qw]``, i.e. ``xyzw`` (confirmed by the recording
    team); ``wxyz`` is kept for rigs that store the scalar first."""

    pair: str = "index"
    """``index`` pairs frame i of every stream with frame i of the master clock
    (correct when all streams share one trigger and have equal length).
    ``nearest`` re-pairs by real timestamps. Streams whose length differs from
    the master always fall back to ``nearest``."""

    max_sync_gap_frames: float = 3.0
    """Hard limit on |stream timestamp - master timestamp|, in frames.

    ``mcap2lerobot`` rejects at 1.5 frames, which is too tight for this recorder:
    a device that stalls briefly (one observed 234ms gap in a jaw encoder) leaves
    a handful of ticks served by a stale sample, and rejecting the episode over
    3 frames out of 1472 throws away 72 good seconds. Frames past
    ``warn_sync_gap_frames`` are still reported."""

    warn_sync_gap_frames: float = 1.5
    """Soft limit: frames beyond this are logged, so recording hiccups stay visible."""

    master_clock: str = "left_tcp"
    """Stream key whose sample count defines the episode length."""

    state_layout: tuple[str, ...] = DEFAULT_STATE_LAYOUT
    streams: tuple[StreamRule, ...] = DEFAULT_STREAM_RULES

    vcodec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 23
    gop: int = 30
    """Keyframe interval. Sources arrive with GOPs of 200+ frames, which makes
    LeRobot's random-access reads crawl; TacCap sits near 57."""

    preset: str = "veryfast"
    workers: int = 0
    """Video encoder threads across camera keys; 0 selects half the CPUs."""

    chunks_size: int = 1000
    data_files_size_in_mb: int = 100
    video_files_size_in_mb: int = 200

    codebase_version: str = "v3.0"
    tolerance_s: float = 1e-4

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.quat_order not in QUAT_ORDERS:
            raise ValueError(f"quat_order must be one of {QUAT_ORDERS}, got {self.quat_order!r}")
        if self.pair not in PAIR_MODES:
            raise ValueError(f"pair must be one of {PAIR_MODES}, got {self.pair!r}")
        if self.max_sync_gap_frames <= 0:
            raise ValueError("max_sync_gap_frames must be positive")
        if self.warn_sync_gap_frames <= 0:
            raise ValueError("warn_sync_gap_frames must be positive")
        if self.gop <= 0:
            raise ValueError("gop must be positive")
        self.state_layout = tuple(self.state_layout)
        self.streams = tuple(self.streams)
        if not self.state_layout:
            raise ValueError("state_layout must not be empty")
        if self.master_clock not in self.state_layout:
            raise ValueError(
                f"master_clock {self.master_clock!r} is not in state_layout {self.state_layout}"
            )

    @property
    def max_sync_gap_s(self) -> float:
        return self.max_sync_gap_frames / self.fps

    @property
    def warn_sync_gap_s(self) -> float:
        return self.warn_sync_gap_frames / self.fps

    def state_names(self, kinds: dict[str, str]) -> list[str]:
        """Build the ``observation.state`` field names from the layout.

        Args:
            kinds: Stream key -> kind, as discovered in the source episode.
        """
        names: list[str] = []
        for key in self.state_layout:
            kind = kinds.get(key)
            if kind is None:
                raise ValueError(f"state_layout references {key!r}, which no source stream provides")
            suffixes = POSE_SUFFIXES if kind == POSE else GRIPPER_SUFFIXES
            names.extend(f"{key}.{suffix}" for suffix in suffixes)
        return names

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConverterConfig":
        """Load a config file, falling back to the built-in defaults per field."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConverterConfig":
        raw = dict(raw)
        rules = raw.pop("streams", None)
        if rules is not None:
            raw["streams"] = tuple(
                StreamRule(pattern=item["pattern"], kind=item["kind"], key=item["key"])
                for item in rules
            )
        if "state_layout" in raw:
            raw["state_layout"] = tuple(raw["state_layout"])
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {key: value for key, value in raw.items() if key not in known}
        settings = {key: value for key, value in raw.items() if key in known}
        settings.setdefault("extra", {})
        settings["extra"].update(extra)
        return cls(**settings)

    def with_overrides(self, **overrides: Any) -> "ConverterConfig":
        """Return a copy with the non-``None`` overrides applied."""
        applied = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **applied) if applied else self
