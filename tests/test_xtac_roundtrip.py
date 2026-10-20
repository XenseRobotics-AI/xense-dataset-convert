"""Real XTac/protobuf/video and source-integrity regression tests."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mcap2lerobot.config import ConverterConfig
from mcap2lerobot.converter import McapToLeRobotConverter, build_features, plan_pairing
from mcap2lerobot.xtac import XTacMcapReader
from xense_dataset_convert._lossless import (
    MANIFEST_ATTACHMENT, digest, inventory, preserve_mcap, restore_lerobot,
    restore_mcap, safe_path,
)
from xense_dataset_convert._xtac_export import export_xtac

CAMERA = "observation.images.head_left"
EXTRA = "observation.tracker_confidence"


@pytest.fixture
def dataset(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("mcap_protobuf")
    root = tmp_path / "original"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "empty").mkdir()
    rng = np.random.default_rng(24)
    images = rng.integers(0, 256, (6, 12, 16, 3), dtype=np.uint8)
    with av.open(str(root / "videos/camera.mp4"), "w") as container:
        stream = container.add_stream("libx264rgb", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 12, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast", "g": "1", "bf": "0"}
        for position, image in enumerate(images):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = position, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    table = pd.DataFrame({
        "observation.state": [np.array([index + .125, -index], np.float32) for index in range(6)],
        "action": [np.array([index / 7], np.float64) for index in range(6)],
        EXTRA: [np.array([index, index + 1], np.float32) for index in range(6)],
        "observation.counter": [np.array([2**60 + index], np.uint64) for index in range(6)],
        "timestamp": np.tile(np.arange(3, dtype=np.float32) / 30, 2),
        "frame_index": np.tile(np.arange(3), 2), "index": np.arange(6),
        "episode_index": np.repeat([0, 1], 3), "task_index": [0, 1, 0, 1, 0, 1],
    })
    table.to_parquet(root / "data/frames.parquet", index=False)
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float64", "shape": [1], "names": ["c"]},
        EXTRA: {"dtype": "float32", "shape": [2], "names": ["left", "right"]},
        "observation.counter": {"dtype": "uint64", "shape": [1], "names": ["counter"]},
        CAMERA: {"dtype": "video", "shape": [12, 16, 3], "names": ["height", "width", "channels"],
                 "info": {"video.codec": "h264", "taccap.rectify_config": {"grid": [1, 2, 3]}}},
    }
    info = {"codebase_version": "v3.0", "fps": 30, "robot_type": "test", "total_episodes": 2,
            "total_frames": 6, "features": features, "collection_stack": "custom-collector",
            "data_path": "data/frames.parquet", "video_path": "videos/camera.mp4", "chunks_size": 1000}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/taccap_extrinsics.json").write_bytes(b'{"calibration": [1, 2, 3]}\n')
    (root / "extra.bin").write_bytes(bytes(range(256)))
    (root / "empty.bin").touch()
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    records = []
    for episode in range(2):
        records.append({"episode_index": episode, "length": 3, "tasks": ["任务甲", "task B"],
                        "data/chunk_index": 0, "data/file_index": 0,
                        "dataset_from_index": episode * 3, "dataset_to_index": (episode + 1) * 3,
                        f"videos/{CAMERA}/chunk_index": 0, f"videos/{CAMERA}/file_index": 0,
                        f"videos/{CAMERA}/from_timestamp": episode * .1,
                        f"videos/{CAMERA}/to_timestamp": (episode + 1) * .1})
    pd.DataFrame(records).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["任务甲", "task B"], name="task")).to_parquet(root / "meta/tasks.parquet")

    def files(episode, chunk, dataset_root):
        return {"parquet": root / "data/frames.parquet", "from_index": episode * 3,
                "to_index": (episode + 1) * 3, "videos": {CAMERA: root / "videos/camera.mp4"},
                "video_ranges": {CAMERA: {"from_timestamp": episode * 3 / 30}}}

    def manifest(episode):
        return {"episode_index": episode, "length": 3, "tasks": ["任务甲", "task B"],
                "task_index_to_task": {"0": "任务甲", "1": "task B"}}

    return SimpleNamespace(
        dataset_root=root, data=info, video_keys=[CAMERA], get_total_episodes=lambda: 2,
        is_v3_format=lambda: True, get_episode_files=files, episode_manifest=manifest,
        images=images, table=table,
    )


def test_default_xtac_streams_preserve_vectors_tasks_and_rgb(dataset, tmp_path):
    output = tmp_path / "xtac"
    assert export_xtac(dataset, output)
    reader = XTacMcapReader(ConverterConfig())
    for index, path in enumerate(sorted(output.glob("*.mcap"))):
        episode = reader.read(path)
        assert len(episode.states) == 3
        assert episode.info["fps"] == 30
        assert episode.episode_meta["task_index_to_task"]["0"] == "任务甲"
        assert episode.info["features"][CAMERA]["info"]["taccap.rectify_config"] == {"grid": [1, 2, 3]}
        assert build_features(episode)[EXTRA]["dtype"] == "float32"
        selection = dataset.table.iloc[index * 3:(index + 1) * 3]
        for feature, values in [("observation.state", episode.states), ("action", episode.actions),
                                *episode.extra_vectors.items()]:
            assert np.array_equal(np.stack([sample.value for sample in values]), np.stack(selection[feature]))
        assert [sample.task_index for sample in episode.states] == selection["task_index"].tolist()
        plan = plan_pairing(episode, 30, 1.5)
        frames = list(reader.iter_frames(episode, plan.video_index))
        assert np.array_equal(np.stack([frame[CAMERA] for frame in frames]), dataset.images[index * 3:(index + 1) * 3])


def test_default_reverse_restores_every_source_file_exactly(dataset, tmp_path):
    export_xtac(dataset, tmp_path / "xtac")
    restored = McapToLeRobotConverter([tmp_path / "xtac"], tmp_path / "restored", "local/test").convert()
    assert inventory(restored) == inventory(dataset.dataset_root)


def test_missing_episode_is_rejected_without_partial_output(dataset, tmp_path):
    export_xtac(dataset, tmp_path / "xtac")
    with pytest.raises(ValueError, match="Incomplete"):
        restore_lerobot([tmp_path / "xtac/episode_001.mcap"], tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_corrupted_source_attachment_is_rejected(dataset, tmp_path):
    from mcap.reader import make_reader
    from mcap.writer import Writer

    export_xtac(dataset, tmp_path / "xtac")
    source = tmp_path / "xtac/episode_000.mcap"
    with source.open("rb") as stream:
        reader = make_reader(stream)
        saved = list(reader.iter_attachments())
        messages = list(reader.iter_messages(log_time_order=False))
        metadata = list(reader.iter_metadata())
    with source.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schemas, channels = {}, {}
        for schema, channel, message in messages:
            if schema.id not in schemas:
                schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
            if channel.id not in channels:
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding,
                                                                schemas[schema.id], channel.metadata)
            writer.add_message(channels[channel.id], message.log_time, message.data,
                               message.publish_time, message.sequence)
        for record in metadata:
            writer.add_metadata(record.name, record.metadata)
        changed = False
        for attachment in saved:
            data = attachment.data
            if attachment.name != MANIFEST_ATTACHMENT and not changed:
                data = bytes([data[0] ^ 1]) + data[1:]
                changed = True
            writer.add_attachment(0, 0, attachment.name, attachment.media_type, data)
        writer.finish()
    with pytest.raises(ValueError, match="checksum"):
        restore_lerobot(sorted((tmp_path / "xtac").glob("*.mcap")), tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_edited_xtac_messages_do_not_replay_stale_source(dataset, tmp_path):
    from mcap.reader import make_reader
    from mcap.writer import Writer

    export_xtac(dataset, tmp_path / "xtac")
    source = tmp_path / "xtac/episode_000.mcap"
    with source.open("rb") as stream:
        reader = make_reader(stream)
        messages = list(reader.iter_messages(log_time_order=False))
        metadata, saved = list(reader.iter_metadata()), list(reader.iter_attachments())
    with source.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schemas, channels = {}, {}
        for schema, channel, message in messages:
            if schema.id not in schemas:
                schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
            if channel.id not in channels:
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding,
                                                                schemas[schema.id], channel.metadata)
            data = message.data
            if channel.topic == "train/state":
                payload = json.loads(data)
                payload["values"][0] = 12345
                data = json.dumps(payload).encode()
            writer.add_message(channels[channel.id], message.log_time, data, message.publish_time, message.sequence)
        for record in metadata:
            writer.add_metadata(record.name, record.metadata)
        for attachment in saved:
            writer.add_attachment(0, 0, attachment.name, attachment.media_type, attachment.data)
        writer.finish()
    with pytest.raises(ValueError, match="projection checksum"):
        restore_lerobot(sorted((tmp_path / "xtac").glob("*.mcap")), tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_original_mcap_is_retained_and_stale_projection_is_refused(dataset, tmp_path):
    export_xtac(dataset, tmp_path / "xtac", preserve_source=False)
    inputs = sorted((tmp_path / "xtac").glob("*.mcap"))
    preserve_mcap(inputs, dataset.dataset_root)
    assert restore_mcap(dataset.dataset_root, tmp_path / "raw-return")
    assert [digest(path) for path in inputs] == [digest(path) for path in sorted((tmp_path / "raw-return").glob("*.mcap"))]
    (dataset.dataset_root / "extra.bin").write_bytes(b"edited")
    with pytest.raises(ValueError, match="projection changed"):
        restore_mcap(dataset.dataset_root, tmp_path / "stale-return")
    assert not (tmp_path / "stale-return").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "x/../escape", "C:/escape", "x\\escape", "./x"])
def test_source_paths_cannot_escape_output(name):
    with pytest.raises(ValueError, match="source path"):
        safe_path(name)


def test_subset_requires_explicit_projection_only_mode(dataset, tmp_path):
    with pytest.raises(ValueError, match="all episodes"):
        export_xtac(dataset, tmp_path / "partial", [1])
    assert not (tmp_path / "partial").exists()
    assert export_xtac(dataset, tmp_path / "partial", [1], preserve_source=False)


def test_camera_failure_cancels_export_without_publishing(dataset, tmp_path):
    original_files = dataset.get_episode_files

    def files(*args):
        value = original_files(*args)
        value["videos"][CAMERA] = tmp_path / "missing.mp4"
        return value

    dataset.get_episode_files = files
    with pytest.raises(FileNotFoundError, match="Missing video"):
        export_xtac(dataset, tmp_path / "broken", intra_workers=2)
    assert not (tmp_path / "broken").exists()


def test_rgb_writer_preserves_pixels_without_source_restore(dataset, tmp_path):
    pytest.importorskip("lerobot")
    import av

    # Multiple cameras exercise LeRobot's process pool (including encoder
    # configuration pickling), rather than only its single-camera fast path.
    other = "observation.images.auxiliary"
    dataset.video_keys.append(other)
    dataset.data["features"][other] = dataset.data["features"][CAMERA]
    original_files = dataset.get_episode_files

    def files(*args):
        value = original_files(*args)
        value["videos"][other] = value["videos"][CAMERA]
        value["video_ranges"][other] = value["video_ranges"][CAMERA]
        return value

    dataset.get_episode_files = files
    export_xtac(dataset, tmp_path / "supported", preserve_source=False)
    converted = McapToLeRobotConverter(
        [tmp_path / "supported"], tmp_path / "projection", "local/test", ConverterConfig(preserve_source=False),
    ).convert()
    for key in (CAMERA, other):
        frames = []
        for video in sorted(converted.glob(f"videos/{key}/**/*.mp4")):
            with av.open(str(video)) as container:
                frames.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
        assert np.array_equal(np.stack(frames), dataset.images)
    frame = pd.concat([pd.read_parquet(path) for path in sorted(converted.glob("data/**/*.parquet"))])
    assert np.array_equal(np.stack(frame[EXTRA]), np.stack(dataset.table[EXTRA]))
    assert np.array_equal(np.stack(frame["observation.counter"]).reshape(-1, 1),
                          np.stack(dataset.table["observation.counter"]))
    info = json.loads((converted / "meta/info.json").read_text())
    assert info["collection_stack"] == "custom-collector"
    assert info["features"][CAMERA]["info"]["taccap.rectify_config"] == {"grid": [1, 2, 3]}


def test_public_facade_defaults_roundtrip_all_original_files(dataset, tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("tabular2mcap")
    pytest.importorskip("spdlog")
    from xense_dataset_convert import Format, convert

    exported = convert(Format.LEROBOT, Format.MCAP, inputs=dataset.dataset_root, output=tmp_path / "mcap")
    restored = convert(Format.MCAP, Format.LEROBOT, inputs=exported.output_path, output=tmp_path / "return")
    assert inventory(restored.output_path) == inventory(dataset.dataset_root)


def test_converter_class_defaults_to_xtac(dataset, tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("tabular2mcap")
    pytest.importorskip("spdlog")
    from lerobot2mcap.converter import LeRobotConverter

    converter = LeRobotConverter(dataset.dataset_root, Path("unused-in-xtac.yaml"))
    assert converter.convert(tmp_path / "xtac")
    restored = McapToLeRobotConverter([tmp_path / "xtac"], tmp_path / "restored", "local/test").convert()
    assert inventory(restored) == inventory(dataset.dataset_root)


@pytest.mark.parametrize("other", ["observation.images.right_wrist", "observation.images.auxiliary"])
def test_parallel_cameras_preserve_frame_order_and_pixels(dataset, tmp_path, other):
    dataset.video_keys.append(other)
    dataset.data["features"][other] = dataset.data["features"][CAMERA]
    original_files = dataset.get_episode_files

    def files(*args):
        value = original_files(*args)
        value["videos"][other] = value["videos"][CAMERA]
        value["video_ranges"][other] = value["video_ranges"][CAMERA]
        return value

    dataset.get_episode_files = files
    export_xtac(dataset, tmp_path / "xtac", intra_workers=2)
    reader = XTacMcapReader(ConverterConfig())
    for index, path in enumerate(sorted((tmp_path / "xtac").glob("*.mcap"))):
        episode = reader.read(path)
        plan = plan_pairing(episode, 30, 1.5)
        frames = list(reader.iter_frames(episode, plan.video_index))
        for key in (CAMERA, other):
            assert np.array_equal(np.stack([frame[key] for frame in frames]), dataset.images[index * 3:(index + 1) * 3])


def test_default_mcap_lerobot_mcap_preserves_unknown_channels(dataset, tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("tabular2mcap")
    pytest.importorskip("spdlog")
    import av

    from mcap.reader import make_reader
    from mcap.writer import Writer
    from xense_dataset_convert import Format, convert

    del dataset.data["features"]["observation.counter"]
    export_xtac(dataset, tmp_path / "raw", preserve_source=False)
    # Add a raw sensor topic the frame-table projection does not understand.
    source = tmp_path / "raw/episode_000.mcap"
    with source.open("rb") as stream:
        reader = make_reader(stream)
        messages = list(reader.iter_messages(log_time_order=False))
        metadata = list(reader.iter_metadata())
    with source.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schemas, channels = {}, {}
        for schema, channel, message in messages:
            if schema.id not in schemas:
                schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
            if channel.id not in channels:
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding,
                                                                schemas[schema.id], channel.metadata)
            writer.add_message(channels[channel.id], message.log_time, message.data,
                               message.publish_time, message.sequence)
        channel = writer.register_channel("imu/extra", "json", 0)
        writer.add_message(channel, 123, b'{"unknown": [1, 2, 3]}', 122, 44)
        for record in metadata:
            writer.add_metadata(record.name, record.metadata)
        writer.finish()
    converted = convert(Format.MCAP, Format.LEROBOT, inputs=tmp_path / "raw", output=tmp_path / "projection")
    # Verify the usable training projection as well as the retained source:
    # returning the original MCAP alone would conceal a broken frame writer.
    table = pd.concat([pd.read_parquet(path) for path in sorted(converted.output_path.glob("data/**/*.parquet"))])
    for key in ("observation.state", "action", EXTRA, "episode_index", "frame_index", "timestamp"):
        expected = np.stack(dataset.table[key])
        # LeRobot stores shape-(1,) numeric features as scalar parquet columns.
        np.testing.assert_array_equal(np.stack(table[key]).reshape(expected.shape), expected)
    tasks = pd.read_parquet(converted.output_path / "meta/tasks.parquet")
    assert [tasks.index[int(index)] for index in table["task_index"]] == [
        "任务甲" if index == 0 else "task B" for index in dataset.table["task_index"]
    ]
    images = []
    for path in sorted(converted.output_path.glob(f"videos/{CAMERA}/**/*.mp4")):
        with av.open(str(path)) as container:
            images.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
    np.testing.assert_array_equal(np.stack(images), dataset.images)
    restored = convert(Format.LEROBOT, Format.MCAP, inputs=converted.output_path, output=tmp_path / "raw-return")
    original = sorted((tmp_path / "raw").glob("*.mcap"))
    returned = sorted(restored.output_path.glob("*.mcap"))
    assert [path.name for path in original] == [path.name for path in returned]
    assert [digest(path) for path in original] == [digest(path) for path in returned]
