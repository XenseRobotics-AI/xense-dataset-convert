"""Pack XTac training MCAPs into a LeRobot v3 dataset, matching taccap-core's Rust exporter.

Input: one or more ``*.train.mcap`` written by the collector's **LeRobot** export with the
training-MCAP sidecar enabled (``xense_train_mcap.timing == "fixed_rate_grid"``, core
``2e43d63`` or later). Everything numeric -- interpolated poses, camera times, the head
origin, tracker confidence, image statistics -- is taken from the MCAP as written; this module
only packs. Nothing is decoded or re-encoded: each ``/camera/*/h264`` access unit is muxed into
the mp4 byte for byte.

The reference is ``taccap-lerobot`` (``DatasetBuilder`` + ``video_native.rs``); the format spec
is ``~/taccap-cross/lerobot-export-spec.md``. Section numbers below (``§n``) point into it.

Deliberately **not** reproduced byte for byte: parquet files (pyarrow and parquet-rs choose
different encodings); the comparison there is logical (schema + values).
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .xtac import XTAC_VIDEO_TOPICS
from .xtac_sidecar import compact_collection_devices, compact_taccap_extrinsics

# Video keys in the order taccap-lerobot uses for meta/episodes columns (bi_all_video_keys).
VIDEO_KEY_ORDER = (
    "observation.images.left_wrist",
    "observation.images.left_tactile_left",
    "observation.images.left_tactile_right",
    "observation.images.right_wrist",
    "observation.images.right_tactile_left",
    "observation.images.right_tactile_right",
    "observation.images.head_left",
    "observation.images.head_right",
)
TOPIC_TO_KEY = {"/" + topic: key for topic, key in XTAC_VIDEO_TOPICS.items()}

FPS = 30
CHUNKS_SIZE = 1000
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200
TRACKER_CONFIDENCE_DIM = 2
TRACKER_CONFIDENCE_NAMES = ["left_tracker.confidence", "right_tracker.confidence"]
COLLECTION_STACK = "taccap-collector"
FIXED_RATE_GRID = "fixed_rate_grid"

# meta/tasks.parquet schema metadata, copied verbatim from taccap-lerobot parquet_io.rs.
TASKS_PANDAS_META = (
    '{"index_columns": ["task"], "column_indexes": [{"name": null, "field_name": null, '
    '"pandas_type": "unicode", "numpy_type": "object", "metadata": {"encoding": "UTF-8"}}], '
    '"columns": [{"name": "task_index", "field_name": "task_index", "pandas_type": "int64", '
    '"numpy_type": "int64", "metadata": null}, {"name": "task", "field_name": "task", '
    '"pandas_type": "unicode", "numpy_type": "object", "metadata": null}], "attributes": {}, '
    '"creator": {"library": "pyarrow", "version": "24.0.0"}, "pandas_version": "2.3.3"}'
)

# capture_mode -> dataset-name prefix (taccap-proto dataset_naming.rs mode_names).
MODE_PREFIX = {"dual_claw": "taccap-g1", "dual_claw_head": "xtac-umi-g1", "dual_claw_head_right": "xtac-umi-g1"}
BEIJING = timezone(timedelta(hours=8))


class TrainMcapError(ValueError):
    """A training MCAP cannot be packed faithfully."""


# ── reading ──────────────────────────────────────────────────────────────────


@dataclass
class TrainEpisode:
    path: Path
    episode_id: str
    started_at_ms: int
    capture_mode: str
    task_name: str
    prompt: str
    robot_type: str
    names: list[str]
    rows: int
    state: np.ndarray                     # (rows, D) float32
    action: np.ndarray                    # (rows, D) float32
    confidence: np.ndarray                # (rows, 2) float32
    videos: dict[str, list[bytes]]        # video_key -> Annex-B access units, grid order
    channel_metadata: dict[str, dict[str, str]]
    calibration: dict[str, Any]           # xumi.camera_calibration.v1 cameras (by topic)
    video_stats: dict[str, dict[str, Any]]  # by h264 topic
    devices_entry: dict[str, Any]
    extrinsics_entry: dict[str, Any]
    tcp_frame_convention: str | None = None   # core >= 556dd3d declares "mirrored-v1"; older MCAPs omit it
    headset_mount: str | None = None          # core >= 35e46e8: "head" / "neck"; older MCAPs omit it

    @property
    def group(self) -> str:
        day = datetime.fromtimestamp(self.started_at_ms / 1000, BEIJING).strftime("%y%m%d")
        return f"{MODE_PREFIX.get(self.capture_mode, '?')}-{self.task_name}-{day}"


def _compressed_video_data(payload: bytes) -> bytes:
    """``foxglove.CompressedVideo.data`` (field 3) from a protobuf message, without protobuf."""
    i, out = 0, None
    while i < len(payload):
        key, i = _varint(payload, i)
        number, wire = key >> 3, key & 7
        if wire == 2:
            length, i = _varint(payload, i)
            if number == 3:
                out = payload[i:i + length]
            i += length
        elif wire == 0:
            _, i = _varint(payload, i)
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            raise TrainMcapError(f"unsupported protobuf wire type {wire}")
    if out is None:
        raise TrainMcapError("CompressedVideo without data")
    return bytes(out)


def _varint(buffer: bytes, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        byte = buffer[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, i
        shift += 7


def read_train_mcap(path: Path) -> TrainEpisode:
    from mcap.reader import make_reader

    path = Path(path)
    with path.open("rb") as handle:
        reader = make_reader(handle)
        metadata = {record.name: dict(record.metadata) for record in reader.iter_metadata()}
        train = metadata.get("xense_train_mcap") or {}
        timing = train.get("timing")
        if timing != FIXED_RATE_GRID:
            raise TrainMcapError(
                f"{path.name}: xense_train_mcap.timing={timing!r}. Only fixed_rate_grid training MCAPs "
                "(collector LeRobot export) map 1:1 onto LeRobot; reference_camera_timestamps rows follow "
                "the reference camera's jittered clock and taccap-lerobot itself refuses to retime them "
                "as fixed-FPS LeRobot (spec §8)."
            )
        missing = [name for name in ("episode", "xense_lerobot_video_stats", "xumi_lerobot_episode")
                   if name not in metadata]
        if "robot_type" not in train:
            missing.append("xense_train_mcap.robot_type")
        if missing:
            raise TrainMcapError(f"{path.name}: missing {missing} -- written by taccap-core before 2e43d63; "
                                 "re-export with a current collector")
        if int(train["fps"]) != FPS:
            raise TrainMcapError(f"{path.name}: fps={train['fps']}, LeRobot export is fixed at {FPS}")
        episode = metadata["episode"]
        summary = reader.get_summary()
        topics = {channel.topic: channel for channel in summary.channels.values()}
        if "/train/tracker_confidence" not in topics:
            raise TrainMcapError(f"{path.name}: no /train/tracker_confidence channel (core before 2e43d63)")
        video_topics = [topic for topic in topics if topic in TOPIC_TO_KEY]
        wanted = ["/train/state", "/train/action", "/train/tracker_confidence", *video_topics]
        rows: dict[str, list] = {topic: [] for topic in wanted}
        for _, channel, message in reader.iter_messages(topics=wanted, log_time_order=True):
            if channel.topic in TOPIC_TO_KEY:
                rows[channel.topic].append(_compressed_video_data(message.data))
            else:
                rows[channel.topic].append(json.loads(message.data)["values"])

    count = int(train["rows"])
    for topic, values in rows.items():
        if len(values) != count:
            raise TrainMcapError(f"{path.name}: {topic} has {len(values)} messages, xense_train_mcap.rows={count}")
    names = train["dimension_names"].split(",")
    if len(names) != int(train["state_dim"]):
        raise TrainMcapError(f"{path.name}: {len(names)} dimension names for state_dim={train['state_dim']}")
    calibration = json.loads(episode.get("xumi.camera_calibration.v1") or "{}").get("cameras") or {}
    sidecar = metadata["xumi_lerobot_episode"]
    return TrainEpisode(
        path=path, episode_id=episode.get("episode_id") or path.name.removesuffix(".train.mcap"),
        started_at_ms=int(episode.get("started_at_ms") or 0), capture_mode=episode.get("capture_mode", ""),
        task_name=episode.get("task_name", ""), prompt=episode.get("prompt", ""),
        robot_type=train["robot_type"], names=names, rows=count,
        tcp_frame_convention=train.get("tcp_frame_convention"),
        headset_mount=train.get("headset_mount"),
        state=np.asarray(rows["/train/state"], dtype=np.float32).reshape(count, len(names)),
        action=np.asarray(rows["/train/action"], dtype=np.float32).reshape(count, len(names)),
        confidence=np.asarray(rows["/train/tracker_confidence"], dtype=np.float32).reshape(count, 2),
        videos={TOPIC_TO_KEY[topic]: rows[topic] for topic in video_topics},
        channel_metadata={TOPIC_TO_KEY[topic]: dict(topics[topic].metadata) for topic in video_topics},
        calibration=calibration,
        video_stats=(json.loads(metadata["xense_lerobot_video_stats"]["json"]).get("topics") or {}),
        devices_entry=json.loads(sidecar["xumi_collection_devices"]),
        extrinsics_entry=json.loads(sidecar["taccap_extrinsics"]),
    )


def order_episodes(episodes: list[TrainEpisode]) -> list[TrainEpisode]:
    """Collector order (§9): ``(started_at, id)`` ascending; one dataset = one naming group."""
    ordered = sorted(episodes, key=lambda e: (e.started_at_ms, e.episode_id))
    groups = sorted({e.group for e in ordered})
    if len(groups) > 1:
        raise TrainMcapError(f"inputs span {len(groups)} datasets {groups}; the collector would export "
                             "one dataset per {prefix}-{task}-{Beijing YYMMDD} -- pack each group separately")
    # mixing TCP conventions in one dataset flips half the left-hand poses without any visible sign
    # same for the headset mount: head.* and the TCP origin mean the head in one, the chest in the other
    for attribute in ("prompt", "robot_type", "names", "tcp_frame_convention", "headset_mount"):
        values = {json.dumps(getattr(e, attribute)) for e in ordered}
        if len(values) > 1:
            raise TrainMcapError(f"inputs disagree on {attribute}: {sorted(values)}")
    keys = {tuple(sorted(e.videos)) for e in ordered}
    if len(keys) > 1:
        raise TrainMcapError(f"inputs carry different video streams: {sorted(keys)}")
    return ordered


# ── H.264 / mp4 (video_native.rs) ─────────────────────────────────────────────


def annex_b_nal_units(access_unit: bytes) -> list[bytes]:
    """Split on 00 00 01 / 00 00 00 01; strip each NAL's trailing zero bytes; drop empties."""
    starts: list[tuple[int, int]] = []
    i, n = 0, len(access_unit)
    while i + 3 <= n:
        if access_unit[i:i + 3] == b"\x00\x00\x01":
            starts.append((i, 3))
            i += 3
        elif i + 4 <= n and access_unit[i:i + 4] == b"\x00\x00\x00\x01":
            starts.append((i, 4))
            i += 4
        else:
            i += 1
    if not starts:
        raise TrainMcapError("missing Annex-B start code")
    units = []
    for position, (start, prefix) in enumerate(starts):
        begin = start + prefix
        end = starts[position + 1][0] if position + 1 < len(starts) else n
        while end > begin and access_unit[end - 1] == 0:
            end -= 1
        if begin < end:
            units.append(access_unit[begin:end])
    if not units:
        raise TrainMcapError("Annex-B access unit contains no NAL units")
    return units


class _Bits:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def bit(self) -> int:
        byte = self.data[self.offset >> 3]
        value = (byte >> (7 - (self.offset & 7))) & 1
        self.offset += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value

    def ue(self) -> int:
        zeros = 0
        while self.bit() == 0:
            zeros += 1
        return (1 << zeros) - 1 + self.bits(zeros)

    def se(self) -> int:
        value = self.ue()
        return (value + 1) // 2 if value & 1 else -(value // 2)


def sps_dimensions(sps: bytes) -> tuple[int, int]:
    """(width, height) from an SPS NAL, following video_native.rs parse_sps_dimensions."""
    rbsp, zeros = bytearray(), 0
    for byte in sps[1:]:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        rbsp.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    bits = _Bits(bytes(rbsp))
    profile = bits.bits(8)
    bits.bits(16)
    bits.ue()
    chroma, separate = 1, False
    if profile in (44, 83, 86, 100, 110, 118, 122, 128, 134, 135, 138, 139, 244):
        chroma = bits.ue()
        if chroma == 3:
            separate = bool(bits.bit())
        bits.ue()
        bits.ue()
        bits.bit()
        if bits.bit():
            for index in range(12 if chroma == 3 else 8):
                if bits.bit():
                    last = nxt = 8
                    for _ in range(16 if index < 6 else 64):
                        if nxt != 0:
                            nxt = (last + bits.se() + 256) % 256
                        if nxt != 0:
                            last = nxt
    bits.ue()
    poc = bits.ue()
    if poc == 0:
        bits.ue()
    elif poc == 1:
        bits.bit()
        bits.se()
        bits.se()
        for _ in range(bits.ue()):
            bits.se()
    bits.ue()
    bits.bit()
    width_mbs, height_units = bits.ue() + 1, bits.ue() + 1
    frame_mbs_only = bits.bit()
    if not frame_mbs_only:
        bits.bit()
    bits.bit()
    crop = (bits.ue(), bits.ue(), bits.ue(), bits.ue()) if bits.bit() else (0, 0, 0, 0)
    factor = 1 if frame_mbs_only else 2
    array_type = 0 if separate else chroma
    sub_w, sub_h = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}[array_type]
    unit_x = 1 if array_type == 0 else sub_w
    unit_y = factor if array_type == 0 else sub_h * factor
    return width_mbs * 16 - (crop[0] + crop[1]) * unit_x, height_units * 16 * factor - (crop[2] + crop[3]) * unit_y


def _box(kind: bytes, payload: bytes) -> bytes:
    if len(payload) + 8 <= 0xFFFFFFFF:
        return struct.pack(">I", len(payload) + 8) + kind + payload
    return struct.pack(">I", 1) + kind + struct.pack(">Q", len(payload) + 16) + payload


def _full(version: int, flags: int) -> bytes:
    return struct.pack(">I", (version << 24) | (flags & 0xFFFFFF))


_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


@dataclass
class _Prepared:
    samples: list[list[bytes]]            # per frame: NALs kept in the sample (no SPS/PPS)
    sync: list[bool]
    sps: list[bytes]
    pps: list[bytes]
    width: int
    height: int
    segments: list[tuple[int, int]]       # (first_frame, frame_count) per episode


def _prepare(segments: list[list[bytes]]) -> _Prepared:
    samples, sync, ranges = [], [], []
    parameter_sets: tuple[set, set] | None = None
    dims = None
    first = 0
    for segment in segments:
        ranges.append((first, len(segment)))
        first += len(segment)
        seg_sps, seg_pps = set(), set()
        for index, access_unit in enumerate(segment):
            kept, is_sync, vcl = [], False, False
            for nal in annex_b_nal_units(access_unit):
                kind = nal[0] & 0x1F
                if 1 <= kind <= 5:
                    vcl, is_sync = True, is_sync or kind == 5
                elif kind == 7:
                    parsed = sps_dimensions(nal)
                    if dims is not None and parsed != dims:
                        raise TrainMcapError(f"H.264 dimensions changed from {dims} to {parsed}")
                    dims = parsed
                    seg_sps.add(nal)
                elif kind == 8:
                    seg_pps.add(nal)
                if kind not in (7, 8):
                    kept.append(nal)
            if not vcl:
                raise TrainMcapError(f"access unit {index} has no VCL NAL")
            if index == 0 and not is_sync:
                raise TrainMcapError("each episode segment must begin with an IDR frame")
            samples.append(kept)
            sync.append(is_sync)
        if len(seg_sps) != 1 or len(seg_pps) != 1:
            raise TrainMcapError("each episode segment must carry exactly one stable SPS and one stable PPS")
        if parameter_sets is not None and parameter_sets != (seg_sps, seg_pps):
            raise TrainMcapError("H.264 SPS/PPS changed between episode segments")
        parameter_sets = (seg_sps, seg_pps)
    if not samples or parameter_sets is None or dims is None:
        raise TrainMcapError("no access units")
    return _Prepared(samples, sync, sorted(parameter_sets[0]), sorted(parameter_sets[1]),
                     dims[0], dims[1], ranges)


def mux_mp4(segments: list[list[bytes]], path: Path, fps: int = FPS) -> _Prepared:
    """ftyp → mdat → moov, timescale = fps, one chunk, CFR (video_native.rs write_prepared_mp4)."""
    prepared = _prepare(segments)
    sizes = [sum(4 + len(nal) for nal in sample) for sample in prepared.samples]
    count = len(sizes)
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41")
    payload = sum(sizes)
    mdat_header = (struct.pack(">I", payload + 8) + b"mdat" if payload <= 0xFFFFFFFF - 8
                   else struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", payload + 16))
    offset = len(ftyp) + len(mdat_header)

    sps0 = prepared.sps[0]
    avcc = bytes([1, sps0[1], sps0[2], sps0[3], 0xFF, 0xE0 | len(prepared.sps)])
    for sps in prepared.sps:
        avcc += struct.pack(">H", len(sps)) + sps
    avcc += bytes([len(prepared.pps)])
    for pps in prepared.pps:
        avcc += struct.pack(">H", len(pps)) + pps
    avc1 = (b"\x00" * 6 + struct.pack(">HHH", 1, 0, 0) + b"\x00" * 12
            + struct.pack(">HHIIIH", prepared.width, prepared.height, 0x480000, 0x480000, 0, 1)
            + b"\x00" * 32 + struct.pack(">Hh", 0x18, -1) + _box(b"avcC", avcc))
    stsd = _box(b"stsd", _full(0, 0) + struct.pack(">I", 1) + _box(b"avc1", avc1))
    stts = _box(b"stts", _full(0, 0) + struct.pack(">III", 1, count, 1))
    syncs = [i + 1 for i, flag in enumerate(prepared.sync) if flag]
    stss = _box(b"stss", _full(0, 0) + struct.pack(f">I{len(syncs)}I", len(syncs), *syncs))
    stsc = _box(b"stsc", _full(0, 0) + struct.pack(">IIII", 1, 1, count, 1))
    stsz = _box(b"stsz", _full(0, 0) + struct.pack(f">II{count}I", 0, count, *sizes))
    stco = (_box(b"stco", _full(0, 0) + struct.pack(">II", 1, offset)) if offset <= 0xFFFFFFFF
            else _box(b"co64", _full(0, 0) + struct.pack(">IQ", 1, offset)))
    stbl = _box(b"stbl", stsd + stts + stss + stsc + stsz + stco)
    dinf = _box(b"dinf", _box(b"dref", _full(0, 0) + struct.pack(">I", 1) + _box(b"url ", _full(0, 1))))
    minf = _box(b"minf", _box(b"vmhd", _full(0, 1) + b"\x00" * 8) + dinf + stbl)
    mdhd = _box(b"mdhd", _full(0, 0) + struct.pack(">IIIIHH", 0, 0, fps, count, 0x55C4, 0))
    hdlr = _box(b"hdlr", _full(0, 0) + struct.pack(">I", 0) + b"vide" + b"\x00" * 12 + b"VideoHandler\x00")
    tkhd = _box(b"tkhd", _full(0, 7) + struct.pack(">IIIII", 0, 0, 1, 0, count) + b"\x00" * 8
                + struct.pack(">HHHH", 0, 0, 0, 0) + _MATRIX
                + struct.pack(">II", prepared.width << 16, prepared.height << 16))
    mvhd = _box(b"mvhd", _full(0, 0) + struct.pack(">IIIIIHH", 0, 0, fps, count, 0x10000, 0x100, 0)
                + b"\x00" * 8 + _MATRIX + b"\x00" * 24 + struct.pack(">I", 2))
    moov = _box(b"moov", mvhd + _box(b"trak", tkhd + _box(b"mdia", mdhd + hdlr + minf)))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as out:
        out.write(ftyp)
        out.write(mdat_header)
        for sample in prepared.samples:
            for nal in sample:
                out.write(struct.pack(">I", len(nal)))
                out.write(nal)
        out.write(moov)
    os.replace(temporary, path)
    return prepared


# ── JSON as serde_json::to_string_pretty writes it ────────────────────────────


def format_f64(value: float) -> str:
    """serde_json's float text: shortest round-trip digits, fixed for -5 < k <= 21, else ``d.ddde-N``."""
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    sign = "-" if value < 0 else ""
    digits_tuple, exponent = Decimal(repr(abs(value))).as_tuple()[1:]   # shortest round-trip digits
    digits = "".join(map(str, digits_tuple)).lstrip("0")
    point = len("".join(map(str, digits_tuple))) + exponent - (len("".join(map(str, digits_tuple))) - len(digits))
    digits = digits.rstrip("0") or "0"                               # value = 0.digits × 10^point
    if -5 < point <= 21:
        if point <= 0:
            return f"{sign}0.{'0' * -point}{digits}"
        if point >= len(digits):
            return f"{sign}{digits}{'0' * (point - len(digits))}.0"
        return f"{sign}{digits[:point]}.{digits[point:]}"
    body = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{body}e{point - 1}"


def dumps_pretty(value: Any, indent: str = "") -> str:
    """``serde_json::to_string_pretty``: 2-space indent, ``"k": v``, ``[]`` / ``{}`` when empty."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_f64(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    inner = indent + "  "
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{json.dumps(str(k), ensure_ascii=False)}: {dumps_pretty(v, inner)}"
                 for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return "[\n" + ",\n".join(inner + dumps_pretty(v, inner) for v in value) + "\n" + indent + "]"
    raise TypeError(f"cannot serialise {type(value).__name__}")


def sort_keys(value: Any) -> Any:
    """What a round trip through ``serde_json::Value`` (BTreeMap, no preserve_order) does to objects."""
    if isinstance(value, dict):
        return {key: sort_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_keys(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_pretty(value), encoding="utf-8")       # no trailing newline (§4.3)


# ── statistics (stats.rs) ──────────────────────────────────────────────────────


def _numeric_stats(rows: np.ndarray) -> dict[str, Any]:
    """Sequential float64 sums (not pairwise) and population std, as NumericAccum."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    if n == 0:
        zeros = [0.0] * values.shape[1]
        return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
    total = np.cumsum(values, axis=0)[-1]
    squares = np.cumsum(values * values, axis=0)[-1]
    mean = total / n
    std = np.sqrt(np.maximum(squares / n - mean * mean, 0.0))
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": mean.tolist(), "std": std.tolist(), "count": [n]}


def _image_stats(per_episode: list[dict[str, Any]]) -> dict[str, Any] | None:
    pixels = frames = 0
    total, squares = [0.0] * 3, [0.0] * 3
    low, high = [math.inf] * 3, [-math.inf] * 3
    merged = False
    for stats in per_episode:
        if not stats.get("pixels") or not stats.get("frames"):
            continue
        merged = True
        pixels += int(stats["pixels"])
        frames += int(stats["frames"])
        for c in range(3):
            total[c] += stats["sum"][c]
            squares[c] += stats["sumsq"][c]
            low[c] = min(low[c], stats["min"][c])
            high[c] = max(high[c], stats["max"][c])
    if not merged:
        return None
    mean = [total[c] / pixels for c in range(3)]
    std = [math.sqrt(max(squares[c] / pixels - mean[c] * mean[c], 0.0)) for c in range(3)]

    def nest(values: list[float]) -> list:
        return [[[v]] for v in values]

    return {"min": nest(low), "max": nest(high), "mean": nest(mean), "std": nest(std), "count": [frames]}


# ── writing ────────────────────────────────────────────────────────────────────


def _next_file(chunk: int, file: int) -> tuple[int, int]:
    return (chunk + 1, 0) if file == CHUNKS_SIZE - 1 else (chunk, file + 1)


def _roll(sizes: Iterable[int], threshold: int) -> list[tuple[int, int]]:
    """Group consecutive episodes into files: flush before an episode that would reach the threshold."""
    out, chunk, file, size, empty = [], 0, 0, 0, True
    for one in sizes:
        if not empty and size + one >= threshold:
            chunk, file = _next_file(chunk, file)
            size, empty = 0, True
        out.append((chunk, file))
        size += one
        empty = False
    return out


@dataclass
class PackResult:
    output: Path
    episodes: int
    frames: int
    video_keys: list[str] = field(default_factory=list)


def pack(inputs: list[Path], output: Path) -> PackResult:
    """Pack training MCAPs into ``output`` (must not exist). Raises :class:`TrainMcapError`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output)
    if output.exists():
        raise TrainMcapError(f"output already exists: {output}")
    episodes = order_episodes([read_train_mcap(Path(p)) for p in inputs])
    if not episodes:
        raise TrainMcapError("no training MCAPs given")
    first = episodes[0]
    dim = len(first.names)
    video_keys = [key for key in VIDEO_KEY_ORDER if key in first.videos]
    starts = np.cumsum([0] + [e.rows for e in episodes])
    total_frames = int(starts[-1])

    # data parquet (§2): one row group per file, episodes never split across files.
    est = [e.rows * (2 * dim * 4 + TRACKER_CONFIDENCE_DIM * 4 + 4 + 32) for e in episodes]
    data_loc = _roll(est, DATA_FILES_SIZE_IN_MB * 1024 * 1024)

    def list_type(d: int) -> pa.DataType:
        return pa.list_(pa.field("item", pa.float32(), nullable=True), d)

    data_schema = pa.schema([
        pa.field("observation.state", list_type(dim), nullable=False),
        pa.field("action", list_type(dim), nullable=False),
        pa.field("observation.tracker_confidence", list_type(TRACKER_CONFIDENCE_DIM), nullable=False),
        pa.field("timestamp", pa.float32(), nullable=False),
        *(pa.field(name, pa.int64(), nullable=False)
          for name in ("frame_index", "episode_index", "index", "task_index")),
    ])
    for location in sorted(set(data_loc)):
        members = [i for i, loc in enumerate(data_loc) if loc == location]
        frame = np.concatenate([np.arange(episodes[i].rows) for i in members])

        def fixed(arrays: list[np.ndarray], d: int) -> pa.Array:
            flat = np.concatenate(arrays).astype(np.float32).reshape(-1)
            return pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), d)

        table = pa.Table.from_arrays([
            fixed([episodes[i].state for i in members], dim),
            fixed([episodes[i].action for i in members], dim),
            fixed([episodes[i].confidence for i in members], TRACKER_CONFIDENCE_DIM),
            pa.array(frame.astype(np.float32) / np.float32(FPS), pa.float32()),
            pa.array(frame, pa.int64()),
            pa.array(np.concatenate([np.full(episodes[i].rows, i) for i in members]), pa.int64()),
            pa.array(np.concatenate([starts[i] + np.arange(episodes[i].rows) for i in members]), pa.int64()),
            pa.array(np.zeros(len(frame), dtype=np.int64), pa.int64()),
        ], schema=data_schema)
        path = output / f"data/chunk-{location[0]:03d}/file-{location[1]:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy")

    # videos (§7): each key rolls independently on the AU-pack size.
    video_refs: dict[str, list[tuple[int, int, float, float]]] = {}
    dims: dict[str, tuple[int, int]] = {}
    for key in video_keys:
        sizes = [16 + sum(8 + len(au) for au in e.videos[key]) for e in episodes]
        locations = _roll(sizes, VIDEO_FILES_SIZE_IN_MB * 1024 * 1024)
        refs: list[tuple[int, int, float, float]] = [None] * len(episodes)  # type: ignore[list-item]
        for location in sorted(set(locations)):
            members = [i for i, loc in enumerate(locations) if loc == location]
            path = output / f"videos/{key}/chunk-{location[0]:03d}/file-{location[1]:03d}.mp4"
            prepared = mux_mp4([episodes[i].videos[key] for i in members], path)
            dims[key] = (prepared.width, prepared.height)
            for i, (first_frame, count) in zip(members, prepared.segments):
                start = first_frame / FPS
                refs[i] = (location[0], location[1], start, start + count / FPS)
        video_refs[key] = refs
        for e in episodes:
            md = e.channel_metadata[key]
            if (md.get("width"), md.get("height")) not in ((None, None), (str(dims[key][0]), str(dims[key][1]))):
                raise TrainMcapError(f"{e.path.name}: {key} channel says {md.get('width')}x{md.get('height')}, "
                                     f"SPS says {dims[key][0]}x{dims[key][1]}")

    # meta/episodes (§3)
    columns: dict[str, pa.Array] = {
        "episode_index": pa.array(range(len(episodes)), pa.int64()),
        "tasks": pa.array([[first.prompt]] * len(episodes), pa.list_(pa.field("item", pa.string(), nullable=True))),
        "length": pa.array([e.rows for e in episodes], pa.int64()),
        "data/chunk_index": pa.array([loc[0] for loc in data_loc], pa.int64()),
        "data/file_index": pa.array([loc[1] for loc in data_loc], pa.int64()),
        "dataset_from_index": pa.array(starts[:-1], pa.int64()),
        "dataset_to_index": pa.array(starts[1:], pa.int64()),
    }
    for key in video_keys:
        refs = video_refs[key]
        columns[f"videos/{key}/chunk_index"] = pa.array([r[0] for r in refs], pa.int64())
        columns[f"videos/{key}/file_index"] = pa.array([r[1] for r in refs], pa.int64())
        columns[f"videos/{key}/from_timestamp"] = pa.array([r[2] for r in refs], pa.float64())
        columns[f"videos/{key}/to_timestamp"] = pa.array([r[3] for r in refs], pa.float64())
    columns["meta/episodes/chunk_index"] = pa.array([0] * len(episodes), pa.int64())
    columns["meta/episodes/file_index"] = pa.array([0] * len(episodes), pa.int64())
    schema = pa.schema([pa.field(name, array.type, nullable=False) for name, array in columns.items()])
    path = output / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(list(columns.values()), schema=schema), path, compression="snappy")

    # meta/tasks.parquet (§5)
    tasks_schema = pa.schema([pa.field("task_index", pa.int64(), nullable=False),
                              pa.field("task", pa.string(), nullable=False)],
                             metadata={"pandas": TASKS_PANDAS_META})
    pq.write_table(pa.Table.from_arrays([pa.array([0], pa.int64()), pa.array([first.prompt])], schema=tasks_schema),
                   output / "meta/tasks.parquet", compression="snappy")

    # meta/stats.json (§4): keys alphabetical (BTreeMap)
    frame = np.concatenate([np.arange(e.rows) for e in episodes])
    stats: dict[str, Any] = {
        "observation.state": _numeric_stats(np.concatenate([e.state for e in episodes])),
        "action": _numeric_stats(np.concatenate([e.action for e in episodes])),
        "observation.tracker_confidence": _numeric_stats(np.concatenate([e.confidence for e in episodes])),
        "timestamp": _numeric_stats(frame / float(FPS)),
        "frame_index": _numeric_stats(frame.astype(np.float64)),
        "episode_index": _numeric_stats(np.concatenate([np.full(e.rows, float(i)) for i, e in enumerate(episodes)])),
        "index": _numeric_stats(np.arange(total_frames, dtype=np.float64)),
        "task_index": _numeric_stats(np.zeros(total_frames)),
    }
    for key in video_keys:
        topic = next(t for t, k in TOPIC_TO_KEY.items() if k == key)
        image = _image_stats([e.video_stats[topic] for e in episodes if topic in e.video_stats])
        if image is not None:
            stats[key] = image
    _write_json(output / "meta/stats.json", {key: stats[key] for key in sorted(stats)})

    # meta/info.json (§1): struct order; video keys alphabetical
    features: dict[str, Any] = {}
    for name in ("action", "observation.state"):
        features[name] = {"dtype": "float32", "names": first.names, "shape": [dim]}
    features["observation.tracker_confidence"] = {"dtype": "float32", "names": TRACKER_CONFIDENCE_NAMES,
                                                  "shape": [TRACKER_CONFIDENCE_DIM]}
    for key in sorted(video_keys):
        width, height = dims[key]
        info = {"video.height": height, "video.width": width, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False,
                "video.video_backend": "pyav"}
        rectify = _rectify(episodes, key)
        if rectify is not None:
            info["taccap.rectified"] = True
            info["taccap.rectify_kind"] = rectify[0]
            info["taccap.rectify_config"] = rectify[1]
        features[key] = {"dtype": "video", "shape": [height, width, 3],
                         "names": ["height", "width", "channels"], "info": info}
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for name in ("frame_index", "episode_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    _write_json(output / "meta/info.json", {
        "codebase_version": "v3.0", "robot_type": first.robot_type, "collection_stack": COLLECTION_STACK,
        # passed through as declared, never converted here (the MCAP's poses already follow it)
        **({"tcp_frame_convention": first.tcp_frame_convention} if first.tcp_frame_convention else {}),
        **({"headset_mount": first.headset_mount} if first.headset_mount else {}),
        "total_episodes": len(episodes), "total_frames": total_frames, "total_tasks": 1,
        "chunks_size": CHUNKS_SIZE, "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB, "fps": FPS,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    })

    # sidecars (§6): the training MCAP's own v1 / v3 entries, episode_index rewritten to this
    # dataset, then compacted to the v2 / v4 files the Rust builder writes (sidecar_v2.rs).
    _write_json(output / "meta/xumi_collection_devices.json", sort_keys(compact_collection_devices(
        [{**e.devices_entry, "episode_index": i} for i, e in enumerate(episodes)])))
    _write_json(output / "meta/taccap_extrinsics.json", sort_keys(compact_taccap_extrinsics(
        [{**e.extrinsics_entry, "episode_index": i} for i, e in enumerate(episodes)])))
    return PackResult(output, len(episodes), total_frames, video_keys)


def _rectify(episodes: list[TrainEpisode], key: str) -> tuple[str, Any] | None:
    """First episode carrying a recipe wins (lib.rs). Calibration beats channel metadata.

    Tactile grid recipes are no longer recorded (core 9c476e0); older MCAPs still carry them, so
    tactile streams are skipped here rather than relying on the input. Wrist / head pass through.
    """
    if "_tactile_" in key:
        return None
    topic = next(t for t, k in TOPIC_TO_KEY.items() if k == key)
    for e in episodes:
        calibration = e.calibration.get(topic)
        if calibration is not None:
            if calibration.get("rectified"):
                return calibration["rectify_kind"], sort_keys(calibration.get("rectify_config"))
            continue
        md = e.channel_metadata.get(key) or {}
        recipe = md.get("rectify_config") or md.get("fisheye_rectify_config")
        if md.get("rectify_kind") and recipe is not None:
            try:
                return md["rectify_kind"], sort_keys(json.loads(recipe))
            except json.JSONDecodeError:
                return md["rectify_kind"], recipe
    return None
