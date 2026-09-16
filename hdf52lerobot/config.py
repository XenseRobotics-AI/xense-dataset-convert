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

from .ee_transform import WORLD_FRAMES, TrackerToEef
from .geometry import QUAT_ORDERS
from .imaging import CHANNEL_ORDER, validate_transforms

VIDEO = "video"
POSE = "pose"
GRIPPER = "gripper"
KINDS = (VIDEO, POSE, GRIPPER)

PAIR_MODES = ("index", "nearest", "grid")

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
        transform: Frame transforms applied to this camera on the way out, in
            order -- see :mod:`hdf52lerobot.imaging`. Videos only; a recorder
            that wrote the sensor's own BGR portrait frames needs
            ``[bgr_to_rgb, transpose]`` to match what the capture path
            produces.
        rgb_reference: Name of an entry in ``ConverterConfig.rgb_references``.
            Given one, this camera's overall tone is brought to that reference,
            the way the Xense SDK does at sensor init. Videos only.
    """

    pattern: str
    kind: str
    key: str
    transform: tuple[str, ...] = ()
    rgb_reference: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown stream kind {self.kind!r}, expected one of {KINDS}")
        re.compile(self.pattern)
        object.__setattr__(self, "transform", validate_transforms(self.transform))
        if self.transform and self.kind != VIDEO:
            raise ValueError(
                f"transform is only meaningful for {VIDEO!r} streams, not {self.kind!r}"
            )
        if self.rgb_reference and self.kind != VIDEO:
            raise ValueError(
                f"rgb_reference is only meaningful for {VIDEO!r} streams, not {self.kind!r}"
            )

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
    """Task text for every episode that ``tasks`` does not name."""

    tasks: dict[str, str] = field(default_factory=dict)
    """Per-episode task text, keyed by the recording directory's name.

    A capture schedule usually gives each session its own task, so one string
    for a whole dataset is the exception rather than the rule. Episodes not
    listed here fall back to ``task``. ``--tasks FILE`` loads a mapping without
    editing the profile."""

    quat_order: str = "xyzw"
    """Component order of the tracker's 4 rotation columns. The UMI rig writes
    ``[x, y, z, qx, qy, qz, qw]``, i.e. ``xyzw`` (confirmed by the recording
    team); ``wxyz`` is kept for rigs that store the scalar first."""

    pair: str = "index"
    """``index`` pairs frame i of every stream with frame i of the master clock
    (correct when all streams share one trigger and have equal length).
    ``nearest`` re-pairs by real timestamps. Streams whose length differs from
    the master always fall back to ``nearest``. ``grid`` pairs like ``index``
    and then resamples the recorder's iterations onto a uniform ``fps`` grid by
    nearest neighbour, for recorders whose loop does not run at ``fps``: a tick
    inside a stall repeats the previous iteration, iterations between ticks are
    dropped, and both are reported per stream."""

    keep_source_timestamp: bool = False
    """Also write ``observation.source_timestamp``: the recorder's absolute
    timestamp (master clock, int64 nanoseconds since the Unix epoch) of the
    iteration each frame came from. ``timestamp`` stays ``frame_index / fps``
    as LeRobot requires; this column keeps the real time axis alongside it for
    audits and re-alignment."""

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
    """Encoder. A software encoder (``libx264``, ``libx265``) takes its quality
    from ``crf``; an NVENC encoder (``h264_nvenc``, ``hevc_nvenc``) is driven at
    constant QP instead, and the options are shaped accordingly."""

    pix_fmt: str = "yuv420p"

    crf: int = 23
    """The encoder's constant-quality number: CRF for the software encoders,
    QP for NVENC. The scales differ -- on this corpus's tactile video, NVENC at
    QP 21 matches libx264 at CRF 23 on PSNR -- so a profile that changes the
    encoder changes this with it."""

    video_options: dict[str, str] = field(default_factory=dict)
    """Encoder options merged over the derived ones, for anything the fields
    above do not name (``tune``, ``rc-lookahead``, ``spatial-aq``)."""

    rgb_references: dict[str, dict[str, float]] = field(default_factory=dict)
    """Reference channel tone per sensor model, e.g.::

        rgb_references:
          GSPS01: {red: 142.189, green: 151.411, blue: 165.365}

    A stream rule naming one of these has its frames shifted so the camera's
    own channel means land on the reference -- what the Xense SDK computes at
    init from the unit's reference image (2.1.x ``init_rgb_bias``) and what a
    recording made with 2.0.x lacks, leaving every sensor its own colour cast.
    """

    rgb_bias_frames: int = 30
    """Frames measured at the start of a recording to estimate a camera's tone.

    The SDK measures one frame at init. Thirty is still the untouched gel and
    steadier: across six recordings of one sensor, a mean from the first thirty
    frames and one sampled over the whole recording differed by 0.2 levels."""
    gop: int = 30
    """Keyframe interval. Sources arrive with GOPs of 200+ frames, which makes
    LeRobot's random-access reads crawl; TacCap sits near 57."""

    preset: str = "veryfast"
    workers: int = 0
    """Cameras encoded in parallel within one episode; 0 selects half the CPUs."""

    codec_threads: int = 1
    """Threads *inside* each encoder and decoder. 0 hands the choice to FFmpeg,
    which sizes every instance to the whole machine.

    One conversion runs an encoder and a decoder per camera at once, and a bulk
    run puts several conversions on the same machine, so FFmpeg's default has
    every one of those instances claiming all the cores: 5 jobs x 7 cameras was
    measured at 1100 threads on 32 cores, a third of a million context switches
    a second, and an episode taking 29.9 s that takes 19.4 s with this set to 1.
    Parallelism belongs at the camera and job level, where it is counted."""

    chunks_size: int = 1000
    data_files_size_in_mb: int = 100
    video_files_size_in_mb: int = 200

    world_frame: str = "none"
    """Which world frame the recorded poses are re-expressed in.

    ``none`` writes them as recorded. ``vive_to_xense`` converts SteamVR's Y-up
    world to X forward / Y left / Z up -- correct for any rig whose poses come
    straight off a Vive tracker, wrong for one whose recorder already did it
    (TacCap). Off by default so an existing profile keeps behaving exactly as it
    did; the RDT profile turns it on."""

    pose_transforms: dict[str, TrackerToEef] = field(default_factory=dict)
    """Per pose-stream ``^Tracker T_EE``, keyed by stream key (``left_tcp``).

    A stream with no entry is written as recorded, which is right only when the
    recorder already placed the frame where the name claims. See
    :mod:`hdf52lerobot.ee_transform`."""

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
        if self.world_frame not in WORLD_FRAMES:
            raise ValueError(
                f"world_frame must be one of {sorted(WORLD_FRAMES)}, got {self.world_frame!r}"
            )
        unknown = set(self.pose_transforms) - set(self.state_layout)
        if unknown:
            raise ValueError(
                f"pose_transforms names {sorted(unknown)}, which state_layout does not contain"
            )
        if self.codec_threads < 0:
            raise ValueError("codec_threads must be 0 (auto) or positive")
        if self.rgb_bias_frames <= 0:
            raise ValueError("rgb_bias_frames must be positive")
        for name, reference in self.rgb_references.items():
            missing = [c for c in CHANNEL_ORDER if c not in reference]
            if missing:
                raise ValueError(f"rgb_references[{name!r}] is missing {missing}")
        for rule in self.streams:
            if rule.rgb_reference and rule.rgb_reference not in self.rgb_references:
                raise ValueError(
                    f"stream {rule.key!r} names rgb_reference {rule.rgb_reference!r}, "
                    f"which is not in rgb_references"
                )
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

    @property
    def is_hardware_encoder(self) -> bool:
        return "nvenc" in self.vcodec or "qsv" in self.vcodec or "vaapi" in self.vcodec

    def encoder_options(self) -> dict[str, str]:
        """Options for ``add_stream``, shaped for the encoder in use.

        The software encoders take ``crf`` plus a minimum keyframe interval;
        NVENC has neither and wants an explicit rate-control mode with a QP.
        """
        options = {"g": str(self.gop), "preset": self.preset}
        if self.is_hardware_encoder:
            options.update({"rc": "constqp", "qp": str(self.crf)})
        else:
            options.update({"crf": str(self.crf), "keyint_min": str(self.gop)})
        options.update({key: str(value) for key, value in self.video_options.items()})
        return options

    def world_matrix(self) -> Any:
        """The world rotation to apply, or None for ``world_frame: none``."""
        return WORLD_FRAMES[self.world_frame]

    def pose_transform_for(self, key: str) -> TrackerToEef:
        """This stream's tracker -> EEF transform; identity when unconfigured."""
        return self.pose_transforms.get(key, TrackerToEef())

    def task_for(self, episode_name: str) -> str:
        """The task text for one recording directory."""
        return self.tasks.get(episode_name, self.task)

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
                StreamRule(
                    pattern=item["pattern"],
                    kind=item["kind"],
                    key=item["key"],
                    transform=tuple(item.get("transform") or ()),
                    rgb_reference=item.get("rgb_reference"),
                )
                for item in rules
            )
        if "state_layout" in raw:
            raw["state_layout"] = tuple(raw["state_layout"])
        if "rgb_references" in raw and raw["rgb_references"] is not None:
            raw["rgb_references"] = {
                str(name): {str(k): float(v) for k, v in dict(ref).items()}
                for name, ref in dict(raw["rgb_references"]).items()
            }
        if "video_options" in raw and raw["video_options"] is not None:
            raw["video_options"] = {
                str(key): str(value) for key, value in dict(raw["video_options"]).items()
            }
        if "pose_transforms" in raw and raw["pose_transforms"] is not None:
            raw["pose_transforms"] = {
                str(key): TrackerToEef.from_mapping(dict(value))
                for key, value in dict(raw["pose_transforms"]).items()
            }
        if "tasks" in raw and raw["tasks"] is not None:
            raw["tasks"] = {str(key): str(value) for key, value in dict(raw["tasks"]).items()}
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
