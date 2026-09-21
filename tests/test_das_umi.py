from pathlib import Path

import numpy as np
import pytest

from mcap2lerobot.config import ConverterConfig
from mcap2lerobot.converter import build_features, plan_pairing
from mcap2lerobot.das_ego import detect_mcap_profile
from mcap2lerobot.das_umi import (
    DasUmiEpisodeReader,
    load_das_umi_config,
    normalize_gripper_opening,
    preflight_das_umi,
)


class FakeVideoDecoder:
    def __init__(self, codec: str):
        self.codec = codec

    def decode(self, data: bytes) -> np.ndarray:
        if data == b"bad":
            raise ValueError("bad packet")
        return np.full((2, 3, 3), data[0], dtype=np.uint8)


def _classes():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    field = descriptor_pb2.FieldDescriptorProto
    proto = descriptor_pb2.FileDescriptorProto(
        name="das_umi_test.proto", package="foxglove", syntax="proto3"
    )

    header = proto.message_type.add(name="Header")
    header.field.add(name="sequence_num", number=1, type=field.TYPE_UINT32)
    header.field.add(name="timestamp", number=2, type=field.TYPE_UINT64)
    vector = proto.message_type.add(name="Vector3")
    for number, name in enumerate(("x", "y", "z"), 1):
        vector.field.add(name=name, number=number, type=field.TYPE_DOUBLE)
    quat = proto.message_type.add(name="Quaternion")
    for number, name in enumerate(("x", "y", "z", "w"), 1):
        quat.field.add(name=name, number=number, type=field.TYPE_DOUBLE)
    pose = proto.message_type.add(name="Pose")
    pose.field.add(
        name="position",
        number=1,
        type=field.TYPE_MESSAGE,
        type_name=".foxglove.Vector3",
    )
    pose.field.add(
        name="orientation",
        number=2,
        type=field.TYPE_MESSAGE,
        type_name=".foxglove.Quaternion",
    )

    image = proto.message_type.add(name="CompressedImage")
    image.field.add(
        name="header", number=1, type=field.TYPE_MESSAGE, type_name=".foxglove.Header"
    )
    image.field.add(name="data", number=2, type=field.TYPE_BYTES)
    image.field.add(name="format", number=3, type=field.TYPE_STRING)
    imu = proto.message_type.add(name="IMUMeasurement")
    imu.field.add(
        name="header", number=1, type=field.TYPE_MESSAGE, type_name=".foxglove.Header"
    )
    imu.field.add(
        name="angular_velocity",
        number=2,
        type=field.TYPE_MESSAGE,
        type_name=".foxglove.Vector3",
    )
    imu.field.add(
        name="linear_acceleration",
        number=3,
        type=field.TYPE_MESSAGE,
        type_name=".foxglove.Vector3",
    )
    magnetic = proto.message_type.add(name="MagneticEncoderMeasurement")
    magnetic.field.add(
        name="header", number=1, type=field.TYPE_MESSAGE, type_name=".foxglove.Header"
    )
    magnetic.field.add(name="value", number=2, type=field.TYPE_DOUBLE)
    pose_frame = proto.message_type.add(name="PoseInFrame")
    pose_frame.field.add(
        name="header", number=1, type=field.TYPE_MESSAGE, type_name=".foxglove.Header"
    )
    pose_frame.field.add(
        name="pose", number=2, type=field.TYPE_MESSAGE, type_name=".foxglove.Pose"
    )
    pose_frame.field.add(name="pose_valid", number=3, type=field.TYPE_BOOL)

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.file.append(proto)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    get = lambda name: message_factory.GetMessageClass(
        pool.FindMessageTypeByName(f"foxglove.{name}")
    )
    return descriptor_set.SerializeToString(), {
        name: get(name)
        for name in (
            "CompressedImage",
            "IMUMeasurement",
            "MagneticEncoderMeasurement",
            "PoseInFrame",
        )
    }


def _write(
    path: Path,
    omit: str | None = None,
    corrupt_prefix: bool = False,
    short: bool = False,
    value_offset: float = 0,
):
    from mcap.writer import Writer

    schema_data, cls = _classes()
    base = 1_700_000_000_000_000_000
    camera_topics = [f"/robot0/sensor/camera{i}/compressed" for i in range(6)] + [
        "/robot1/sensor/camera0/compressed",
        "/robot2/sensor/camera0/compressed",
    ]
    imu_topics = [f"/robot{i}/sensor/imu" for i in range(3)]
    magnetic_topics = [f"/robot{i}/sensor/magnetic_encoder" for i in (1, 2)]
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schemas = {
            name: writer.register_schema(f"foxglove.{name}", "protobuf", schema_data)
            for name in cls
        }
        for topic_index, topic in enumerate(camera_topics):
            if topic == omit:
                continue
            channel = writer.register_channel(
                topic, "protobuf", schemas["CompressedImage"]
            )
            offset = (
                topic_index * 100_000
                if topic_index < 6
                else (topic_index - 5) * 1_000_000
            )
            for index in range(2 if short else 7):
                timestamp = base + index * 1_000_000_000 // 30 + offset
                payload = cls["CompressedImage"](
                    data=b"bad"
                    if corrupt_prefix and index == 0
                    else bytes([index + 1]),
                    format="fake",
                )
                payload.header.timestamp = timestamp
                payload.header.sequence_num = index
                writer.add_message(
                    channel, timestamp, payload.SerializeToString(), timestamp, index
                )
        for topic in imu_topics:
            if topic == omit:
                continue
            channel = writer.register_channel(
                topic, "protobuf", schemas["IMUMeasurement"]
            )
            for index in range(7 if short else 41):
                timestamp = base + index * 5_000_000
                payload = cls["IMUMeasurement"]()
                payload.header.timestamp = timestamp
                payload.header.sequence_num = index
                (
                    payload.angular_velocity.x,
                    payload.angular_velocity.y,
                    payload.angular_velocity.z,
                ) = (index, 2, 3)
                (
                    payload.linear_acceleration.x,
                    payload.linear_acceleration.y,
                    payload.linear_acceleration.z,
                ) = (4, 5, 6)
                writer.add_message(
                    channel, timestamp, payload.SerializeToString(), timestamp, index
                )
        for topic in magnetic_topics:
            if topic == omit:
                continue
            channel = writer.register_channel(
                topic, "protobuf", schemas["MagneticEncoderMeasurement"]
            )
            for index in range(2 if short else 11):
                timestamp = base + index * 20_000_000
                payload = cls["MagneticEncoderMeasurement"](
                    value=value_offset + index * 0.0275 - 0.0275
                )
                payload.header.timestamp = timestamp
                payload.header.sequence_num = index
                writer.add_message(
                    channel, timestamp, payload.SerializeToString(), timestamp, index
                )
        for topic in ("/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"):
            if topic == omit:
                continue
            channel = writer.register_channel(topic, "protobuf", schemas["PoseInFrame"])
            for index in range(2 if short else 7):
                timestamp = base + index * 1_000_000_000 // 30
                payload = cls["PoseInFrame"](pose_valid=True)
                payload.header.timestamp = timestamp
                payload.header.sequence_num = index
                payload.pose.position.x = index
                payload.pose.orientation.w = 1
                writer.add_message(
                    channel, timestamp, payload.SerializeToString(), timestamp, index
                )
        writer.finish()


def test_gripper_normalization_clips_scales_and_rounds_to_two_decimals():
    rules = load_das_umi_config(None).gripper_normalization
    values = np.array([-0.01, 0.0, 0.0011, 0.055, 0.11, 0.165, 0.22, 0.25])
    actual = normalize_gripper_opening(values, rules)
    np.testing.assert_array_equal(
        actual,
        np.array([0.0, 0.0, 0.01, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=np.float32),
    )
    assert rules.min_opening_m == 0.0
    assert rules.max_opening_m == 0.220
    assert rules.decimals == 2


def test_das_umi_preflight_grid_state_action_and_streaming(tmp_path):
    path = tmp_path / "merged.mcap"
    _write(path, corrupt_prefix=True)
    assert detect_mcap_profile(path) == "das-umi"
    config = load_das_umi_config(None)
    result = preflight_das_umi(
        path, config, export_action=True, video_decoder_factory=FakeVideoDecoder
    )
    assert result.passed, result.issues
    episode = result.episode
    assert episode is not None
    assert len(episode.videos) == 8
    assert all(video.packet_indices[0] == 1 for video in episode.videos.values())
    assert len(episode.states[0].value) == 11
    assert len(episode.actions) == len(episode.states)
    assert np.array_equal(episode.actions[0].value, result.episode.states[1].value)
    expected_grid = (
        episode.timeline_ns[0]
        + np.arange(len(episode.timeline_ns)) * 1_000_000_000 // 30
    )
    assert np.array_equal(episode.timeline_ns, expected_grid.astype(np.int64))

    features = build_features(episode)
    assert features["observation.state"]["names"][0] == "left_gripper.pos"
    assert features["observation.state"]["names"][-1] == "head_camera.r6"
    assert features["action"]["names"] == features["observation.state"]["names"]
    assert features["observation.imu"]["names"] == [
        f"imu.{name}"
        for name in (
            "angular_velocity.x",
            "angular_velocity.y",
            "angular_velocity.z",
            "linear_acceleration.x",
            "linear_acceleration.y",
            "linear_acceleration.z",
        )
    ]
    assert features["observation.imu.left_wrist"]["names"][0] == (
        "imu.left_wrist.angular_velocity.x"
    )
    assert features["observation.imu.right_wrist"]["names"][0] == (
        "imu.right_wrist.angular_velocity.x"
    )
    assert features["observation.pose.head_relative"]["names"][0] == (
        "pose.head_relative.x"
    )
    assert "observation.pose.head_world" not in features

    # Only the intentional current/next-frame training pair may share dimension
    # names. Auxiliary sensors must never be interpreted as paired curves.
    auxiliary = {
        key: set(spec["names"] or [])
        for key, spec in features.items()
        if key.startswith("observation.")
        and key != "observation.state"
        and spec["dtype"] not in {"video", "image"}
    }
    for key, names in auxiliary.items():
        for other_key, other_names in auxiliary.items():
            if key < other_key:
                assert names.isdisjoint(other_names), (key, other_key)
    assert not any(key.startswith("observation.source_") for key in features)

    plan = plan_pairing(episode, 30, 1.5)
    reader = DasUmiEpisodeReader(video_decoder_factory=FakeVideoDecoder)
    frames = list(reader.iter_frames(episode, plan.video_index))
    assert len(frames) == len(episode.timeline_ns)
    assert frames[0]["observation.images.camera0"][0, 0, 0] == 2


def test_das_umi_schema_v1_retains_legacy_inline_audit(tmp_path):
    path = tmp_path / "merged.mcap"
    _write(path)
    result = preflight_das_umi(
        path,
        load_das_umi_config(None),
        video_decoder_factory=FakeVideoDecoder,
        schema_version=1,
    )
    assert result.passed, result.issues
    features = build_features(result.episode)
    assert "observation.pose.head_world" in features
    assert "observation.source_timestamp.left_magnetic" in features
    assert "observation.source_index.right_magnetic" in features


def test_das_umi_schema_v2_retains_raw_gripper_values(tmp_path):
    path = tmp_path / "merged.mcap"
    _write(path)
    config = load_das_umi_config(None)
    result = preflight_das_umi(
        path,
        config,
        export_action=False,
        video_decoder_factory=FakeVideoDecoder,
        schema_version=2,
    )
    episode = result.episode
    assert episode is not None
    for column, label in enumerate(("left_magnetic", "right_magnetic")):
        data = result.streams[label]
        nearest = np.abs(data.times_ns[:, None] - episode.timeline_ns).argmin(axis=0)
        raw = np.stack(data.values)[nearest, 0]
        np.testing.assert_array_equal(
            np.stack([sample.value for sample in episode.states])[:, column], raw
        )


def test_das_umi_missing_finger_sensor_is_a_reportable_failure(tmp_path):
    pytest.importorskip("mcap_protobuf")
    path = tmp_path / "partial.mcap"
    _write(path, omit="/robot2/sensor/imu")
    assert detect_mcap_profile(path) == "das-umi"
    result = preflight_das_umi(
        path, load_das_umi_config(None), video_decoder_factory=FakeVideoDecoder
    )
    assert not result.passed
    assert any("missing required stream right_imu" in issue for issue in result.issues)
    assert "robot2/sensor/camera0/compressed" in result.inventory


def test_das_umi_pose_uses_matrix_columns_and_default_has_action(tmp_path):
    path = tmp_path / "merged.mcap"
    _write(path)
    result = preflight_das_umi(
        path, load_das_umi_config(None), video_decoder_factory=FakeVideoDecoder
    )
    assert result.passed
    episode = result.episode
    assert episode is not None
    assert episode.actions
    # State is left encoder, right encoder, then identity head rotation. The
    # first two matrix columns flatten to [1, 0, 0, 0, 1, 0].
    assert episode.states[0].value[5:].tolist() == [1, 0, 0, 0, 1, 0]


def test_rejected_preflight_writes_reports_but_not_output(tmp_path):
    from mcap2lerobot.converter import McapToLeRobotConverter
    from mcap2lerobot.das_umi import PreflightError

    path = tmp_path / "partial.mcap"
    _write(path, omit="/robot2/sensor/imu")
    output = tmp_path / "dataset"
    converter = McapToLeRobotConverter(
        [path],
        output,
        "local/test",
        ConverterConfig(profile="das-umi", check_only=True),
    )
    with pytest.raises(PreflightError):
        converter.convert()
    reports = tmp_path / "dataset_reports"
    assert not output.exists()
    assert "Result: FAIL" in (reports / "preflight.md").read_text()
    assert "NOT RUN" in (reports / "conversion.md").read_text()


@pytest.mark.parametrize("enabled", [None, True, False])
def test_training_vectors_and_topic_mapping(tmp_path, enabled):
    import json
    from dataclasses import replace

    from mcap2lerobot.das_umi import (
        topic_mapping,
        verify_alignment_sidecar,
        write_alignment_sidecar,
        write_schema_metadata,
        write_topic_mapping,
    )

    path = tmp_path / "merged.mcap"
    _write(path)
    config = load_das_umi_config(None)
    # Mapping must follow configuration overrides, not hard-coded output names.
    config = replace(
        config,
        streams=tuple(
            replace(rule, feature="observation.imu.custom")
            if rule.key == "ego_imu"
            else rule
            for rule in config.streams
        ),
    )
    full = preflight_das_umi(path, config, False, FakeVideoDecoder).episode
    result = preflight_das_umi(path, config, enabled, FakeVideoDecoder)
    assert result.passed, result.issues
    episode = result.episode
    features = build_features(episode)
    names = ["left_gripper.pos", "right_gripper.pos"] + [
        f"head_camera.{name}"
        for name in ("x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6")
    ]
    assert features["observation.state"] == {
        "dtype": "float32",
        "shape": (11,),
        "names": names,
    }
    all_states = np.stack([s.value for s in full.states])
    expected = all_states if enabled is False else all_states[:-1]
    np.testing.assert_array_equal(np.stack([s.value for s in episode.states]), expected)
    if enabled is False:
        assert not episode.actions and "action" not in features
    else:
        np.testing.assert_array_equal(
            np.stack([s.value for s in episode.actions]), all_states[1:]
        )
        assert features["action"] == features["observation.state"]
    for i, label in enumerate(("left_magnetic", "right_magnetic")):
        data = result.streams[label]
        nearest = np.abs(data.times_ns[:, None] - episode.timeline_ns).argmin(axis=0)
        normalized = normalize_gripper_opening(
            np.stack(data.values)[nearest, 0], config.gripper_normalization
        )
        np.testing.assert_array_equal(expected[:, i], normalized)
        assert f"observation.source_timestamp.{label}" not in features
        assert f"observation.source_index.{label}" not in features
    assert not any(key.startswith("observation.magnetic.") for key in features)
    assert episode.hidden_observations == {
        "observation.magnetic.left.raw_value",
        "observation.magnetic.right.raw_value",
        "observation.pose.head_world",
    }
    mapping = topic_mapping([result])
    write_alignment_sidecar(tmp_path, [episode])
    verify_alignment_sidecar(tmp_path, [episode])
    write_schema_metadata(tmp_path, [episode])
    write_topic_mapping(tmp_path, [result])
    assert (
        json.loads((tmp_path / "meta/das_umi_topic_mapping.json").read_text())
        == mapping
    )
    assert "topic" not in mapping["derived_fields"][0]
    assert mapping["schema_version"] == "3.0.0"
    schema = json.loads((tmp_path / "meta/das_umi_schema.json").read_text())
    assert schema["version"] == "3.0.0"
    assert schema["action"]["source"] == "next_observation_state_proxy"
    assert schema["canonical_head_pose"]["standalone_observation_exported"] is False
    assert schema["gripper_normalization"]["max_opening_m"] == 0.220
    assert schema["gripper_normalization"]["decimals"] == 2
    assert schema["gripper_normalization"]["closed_value"] == 0.0
    assert schema["gripper_normalization"]["open_value"] == 1.0
    assert mapping["derived_fields"][0]["storage"] == ("meta/das_umi_alignment.parquet")
    topics = mapping["episodes"][0]["topics"]
    assert len(topics) == 15
    for topic in topics:
        assert topic["audit_columns"]["storage"] == ("meta/das_umi_alignment.parquet")
        for target in topic["targets"]:
            spec = features[target["field"]]
            for component in target.get("components", []):
                assert spec["names"][component["index"]] == component["name"]
            if target["field"] == "action":
                assert enabled is not False and target["alignment"] == "next_frame"
    assert any(t["field"] == "observation.imu.custom" for t in topics[8]["targets"])
    assert sum(
        t["field"] == "action" for topic in topics for t in topic["targets"]
    ) == (0 if enabled is False else 3)


def test_single_aligned_row_requires_observation_only(tmp_path):
    path = tmp_path / "short.mcap"
    _write(path, short=True)
    config = load_das_umi_config(None)
    result = preflight_das_umi(path, config, video_decoder_factory=FakeVideoDecoder)
    assert not result.passed
    assert "action requires at least two aligned rows" in result.issues
    result = preflight_das_umi(path, config, False, FakeVideoDecoder)
    assert result.passed, result.issues
    assert len(result.episode.states) == 1


@pytest.fixture
def fake_video_pipeline(monkeypatch):
    import mcap2lerobot.das_umi as module

    real_preflight = module.preflight_das_umi
    monkeypatch.setattr(
        module,
        "preflight_das_umi",
        lambda *args, **kwargs: real_preflight(
            *args, **kwargs, video_decoder_factory=FakeVideoDecoder
        ),
    )
    monkeypatch.setattr(
        module, "DasUmiEpisodeReader", lambda: DasUmiEpisodeReader(FakeVideoDecoder)
    )


@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("entrypoint", ["direct", "cli", "facade"])
def test_batch_entrypoints_match_and_never_cross_episodes(
    tmp_path, monkeypatch, fake_video_pipeline, enabled, entrypoint
):
    from test_mcap2lerobot import FakeDataset

    from mcap2lerobot import cli
    from mcap2lerobot.converter import McapToLeRobotConverter
    from xense_dataset_convert import Format, convert

    paths = [tmp_path / f"episode{i}.mcap" for i in range(2)]
    for i, path in enumerate(paths):
        _write(path, value_offset=100 * i)
    seen = []
    original_init = McapToLeRobotConverter.__init__

    def factory(**kwargs):
        dataset = FakeDataset(kwargs)
        seen.append(dataset)
        return dataset

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs, dataset_factory=factory)

    monkeypatch.setattr(McapToLeRobotConverter, "__init__", init)
    output = tmp_path / "dataset"
    options = {} if enabled is None else {"export_action": enabled}
    if entrypoint == "direct":
        McapToLeRobotConverter(
            paths, output, "local/test", ConverterConfig(**options)
        ).convert()
    elif entrypoint == "facade":
        convert(Format.MCAP, Format.LEROBOT, inputs=paths, output=output, **options)
    else:
        args = [str(p) for p in paths] + ["-o", str(output)]
        if enabled is not None:
            args += ["--export-action" if enabled else "--no-export-action"]
        monkeypatch.setattr("sys.argv", ["mcap2lerobot", *args])
        assert cli.main() == 0
    dataset = seen[0]
    assert dataset.finalized and len(dataset.episodes) == 2
    for path, frames in zip(paths, dataset.episodes):
        full = preflight_das_umi(
            path, load_das_umi_config(None), False, FakeVideoDecoder
        ).episode
        assert len(frames) == len(full.states) - int(enabled is not False)
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(
                frame["observation.state"], full.states[i].value
            )
            if enabled is not False:
                np.testing.assert_array_equal(frame["action"], full.states[i + 1].value)
            else:
                assert "action" not in frame
            assert not any(key.startswith("observation.magnetic.") for key in frame)
            assert not any(key.startswith("observation.source_") for key in frame)
            assert "observation.pose.head_world" not in frame
            assert sum(key.startswith("observation.images.") for key in frame) == 8
    report = (tmp_path / "dataset_reports/conversion.md").read_text()
    assert f"Action tail rows dropped: {0 if enabled is False else 2}" in report
    assert (
        "N/A (不适用" in report
        if enabled is False
        else "including final target): PASS" in report
    )
