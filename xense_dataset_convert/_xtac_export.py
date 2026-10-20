"""Export the public XTac JSON/protobuf layout, without lossy video encoding."""

from __future__ import annotations

import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Iterator, Sequence
from urllib.parse import quote

import numpy as np

from ._lossless import add_source_attachments, build_manifest, inventory, restore_mcap, tracked_writer

LEROBOT_METADATA = "xense_lerobot"
logger = logging.getLogger(__name__)


def compressed_video_type() -> Any:
    """Build the standard Foxglove protobuf descriptor without generated files."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2

    file = descriptor_pb2.FileDescriptorProto(
        name="foxglove/CompressedVideo.proto", package="foxglove", syntax="proto3",
        dependency=["google/protobuf/timestamp.proto"],
    )
    message = file.message_type.add(name="CompressedVideo")
    for name, number, kind, type_name in [
        ("timestamp", 1, 11, ".google.protobuf.Timestamp"),
        ("frame_id", 2, 9, ""), ("data", 3, 12, ""), ("format", 4, 9, ""),
    ]:
        field = message.field.add(name=name, number=number, type=kind, label=1)
        if type_name:
            field.type_name = type_name
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("foxglove.CompressedVideo"))


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=lambda item: item.item()).encode("utf-8")


def _video_packets(path: Path, start: float, length: int, fps: int) -> Iterator[bytes]:
    """Decode only the selected range and encode RGB exactly, one packet per frame."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        codec = av.CodecContext.create("libx264rgb", "w")
        codec.width, codec.height = stream.width, stream.height
        codec.pix_fmt = "rgb24"
        codec.time_base = Fraction(1, fps)
        codec.framerate = Fraction(fps)
        codec.thread_count = 1
        codec.options = {"crf": "0", "preset": "ultrafast", "tune": "zerolatency",
                         "g": "1", "bf": "0"}
        container.seek(max(0, int(start / stream.time_base)), stream=stream, backward=True)
        written = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 0.5 / fps < start:
                continue
            target_time = start + written / fps
            if abs(frame.time - target_time) > 0.51 / fps:
                raise ValueError(f"{path}: video time {frame.time} does not match {target_time}")
            rgb = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
            rgb.pts = written
            packets = codec.encode(rgb)
            if len(packets) != 1:
                raise ValueError("Lossless encoder must emit exactly one packet per frame")
            yield bytes(packets[0])
            written += 1
            if written == length:
                break
        if written != length:
            raise ValueError(f"{path}: decoded {written} of {length} frames")
        if codec.encode(None):
            raise ValueError("Unexpected buffered packet from all-keyframe encoder")


def _camera_packets(topics: dict, files: dict, length: int, fps: int,
                    workers: int) -> Iterator[tuple[str, str, int, bytes]]:
    """Encode cameras concurrently, with a bounded queue and one MCAP writer."""
    if not topics:
        return
    queue: Queue = Queue(maxsize=2 * min(workers, len(topics)))
    stop = Event()

    def send(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def produce(topic, key):
        try:
            path = files["videos"].get(key)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing video for {key}")
            start = files.get("video_ranges", {}).get(key, {}).get("from_timestamp", 0.0)
            for position, data in enumerate(_video_packets(path, start, length, fps)):
                if not send((topic, key, position, data)):
                    return
        finally:
            send((topic, key, -1, b""))

    with ThreadPoolExecutor(max_workers=min(workers, len(topics))) as pool:
        futures = {topic: pool.submit(produce, topic, key) for topic, key in topics.items()}
        remaining = len(futures)
        try:
            while remaining:
                try:
                    item = queue.get(timeout=0.1)
                except Empty:
                    continue
                if item[2] == -1:
                    remaining -= 1
                    futures[item[0]].result()
                else:
                    yield item
        finally:
            stop.set()


def export_xtac(dataset: Any, output: Path, episodes: Sequence[int] | None = None,
                preserve_source: bool = True, intra_workers: int = 8) -> bool:
    """Write XTac episodes and, by default, a complete recoverable source snapshot."""
    import pandas as pd
    from mcap_protobuf.schema import register_schema
    from mcap2lerobot.xtac import XTAC_VIDEO_TOPICS

    root, output = dataset.dataset_root.resolve(), Path(output).resolve()
    if intra_workers < 1:
        raise ValueError("intra_workers must be positive")
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source dataset for lossless conversion")
    if preserve_source and (root / "meta/source_mcap/manifest.json").exists():
        if episodes is not None:
            raise ValueError("Episode filtering cannot restore an original MCAP collection")
        return restore_mcap(root, output)
    if output.exists():
        raise FileExistsError(output)
    all_episodes = list(range(dataset.get_total_episodes()))
    selected = all_episodes if episodes is None else list(episodes)
    if not selected or len(selected) != len(set(selected)) or any(ep not in all_episodes for ep in selected):
        raise ValueError("Invalid or duplicate episode selection")
    if preserve_source and set(selected) != set(all_episodes):
        raise ValueError("Lossless source restoration requires all episodes; use --no-preserve-source for a subset")
    manifest = build_manifest(root, selected) if preserve_source else None
    reverse_topics = {feature: topic for topic, feature in XTAC_VIDEO_TOPICS.items()}
    video_type = compressed_video_type()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xense-xtac-", dir=output.parent) as temporary:
        staging = Path(temporary) / "mcap"
        staging.mkdir()
        for episode in selected:
            logger.info("Exporting XTac episode %d (%d episodes selected)", episode, len(selected))
            chunk = episode // int(dataset.data.get("chunks_size", 1000))
            files = dataset.get_episode_files(episode, chunk, root)
            frame = pd.read_parquet(files["parquet"])
            if dataset.is_v3_format():
                frame = frame[(frame["index"] >= files["from_index"]) & (frame["index"] < files["to_index"])]
                episode_meta = dataset.episode_manifest(episode)
            else:
                episode_meta = {"episode_index": episode, "length": len(frame), "tasks": [], "task_index_to_task": {}}
            if frame.empty:
                raise ValueError(f"Episode {episode} has no frames")
            times = np.rint(np.asarray(frame["timestamp"], dtype=np.float64) * 1_000_000_000).astype(np.int64)
            if np.any(times < 0) or np.any(np.diff(times) < 0):
                raise ValueError("XTac timestamps must be nonnegative and monotonic")
            extra_topics = {}
            for feature, spec in dataset.data["features"].items():
                if feature in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index") or feature not in frame.columns:
                    continue
                if spec["dtype"] not in ("video", "image", "string"):
                    extra_topics["lerobot/features/" + quote(feature, safe="")] = feature
            video_topics = {reverse_topics.get(key, "camera/lerobot/" + quote(key, safe="") + "/h264"): key
                            for key in dataset.video_keys}
            path = staging / f"episode_{episode:03d}.mcap"
            with path.open("wb") as stream:
                writer = tracked_writer(stream)
                writer.start(profile="", library="xense-dataset-convert/xtac-v1")
                writer.add_metadata(LEROBOT_METADATA, {
                    "version": "1", "info": _json(dataset.data).decode(),
                    "episode": _json(episode_meta).decode(),
                    "extra_topics": _json(extra_topics).decode(),
                    "video_topics": _json(video_topics).decode(),
                })
                state_spec = dataset.data["features"]["observation.state"]
                writer.add_metadata("xense_train_mcap", {
                    "fps": str(dataset.data["fps"]), "nominal_encoder_fps": str(dataset.data["fps"]),
                    "dimension_names": ",".join(state_spec.get("names") or []),
                    "state_dim": str(np.asarray(frame.iloc[0]["observation.state"]).size),
                })
                writer.add_metadata("episode", {"episode_id": str(episode),
                                               "task_id": str((episode_meta.get("tasks") or [""])[0])})
                schema = writer.register_schema("xense.train.Array", "jsonschema", _json({"type": "object"}))
                json_topics = {"train/state": "observation.state", "train/action": "action", **extra_topics}
                for topic, feature in json_topics.items():
                    channel = writer.register_channel(topic, "json", schema)
                    for position, (_, row) in enumerate(frame.iterrows()):
                        value = np.asarray(row[feature]).tolist()
                        if not isinstance(value, list):
                            value = [value]
                        payload = {"values": value, "task_index": int(row.get("task_index", 0))}
                        stamp = int(times[position])
                        writer.add_message(channel, stamp, _json(payload), stamp, position)
                video_schema = register_schema(writer, video_type)
                video_channels = {}
                for topic, key in video_topics.items():
                    video_channels[topic] = writer.register_channel(topic, "protobuf", video_schema)
                packets = _camera_packets(video_topics, files, len(frame), dataset.data["fps"], intra_workers)
                try:
                    for topic, key, position, data in packets:
                        stamp = int(times[position])
                        message = video_type(frame_id=key, format="h264", data=data)
                        message.timestamp.FromNanoseconds(stamp)
                        writer.add_message(video_channels[topic], stamp, message.SerializeToString(), stamp, position)
                finally:
                    packets.close()
                if manifest is not None:
                    add_source_attachments(writer, root, manifest, episode)
                writer.finish()
            logger.info("Finished XTac episode %d: %d frames, %d bytes", episode, len(frame), path.stat().st_size)
        if manifest is not None:
            expected = {key: manifest[key] for key in ("files", "directories")}
            expected["files"] = [{key: entry[key] for key in ("path", "size", "sha256")} for entry in manifest["files"]]
            if inventory(root) != expected:
                raise ValueError("Source changed during conversion; refusing an inconsistent export")
        staging.rename(output)
    return True
