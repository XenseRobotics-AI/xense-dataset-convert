from pathlib import Path

import numpy as np
import pytest

from mcap2lerobot.config import ConverterConfig
from mcap2lerobot.converter import build_features, plan_pairing
from mcap2lerobot.das_ego import detect_mcap_profile
from mcap2lerobot.das_umi import (
    DasUmiEpisodeReader,
    load_das_umi_config,
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


def _write(path: Path, omit: str | None = None, corrupt_prefix: bool = False):
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
            for index in range(7):
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
            for index in range(41):
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
            for index in range(11):
                timestamp = base + index * 20_000_000
                payload = cls["MagneticEncoderMeasurement"](value=index + 0.25)
                payload.header.timestamp = timestamp
                payload.header.sequence_num = index
                writer.add_message(
                    channel, timestamp, payload.SerializeToString(), timestamp, index
                )
        for topic in ("/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"):
            if topic == omit:
                continue
            channel = writer.register_channel(topic, "protobuf", schemas["PoseInFrame"])
            for index in range(7):
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
    assert features["observation.state"]["names"][0] == "left_magnetic.raw_value"
    assert features["observation.state"]["names"][-1] == "head.r6"
    assert features["action"]["names"] == features["observation.state"]["names"]
    assert "observation.source_timestamp.left_wrist" in features
    assert "observation.source_index.right_wrist" in features

    plan = plan_pairing(episode, 30, 1.5)
    reader = DasUmiEpisodeReader(video_decoder_factory=FakeVideoDecoder)
    frames = list(reader.iter_frames(episode, plan.video_index))
    assert len(frames) == len(episode.timeline_ns)
    assert frames[0]["observation.images.camera0"][0, 0, 0] == 2


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


def test_das_umi_pose_uses_matrix_columns_and_default_has_no_action(tmp_path):
    path = tmp_path / "merged.mcap"
    _write(path)
    result = preflight_das_umi(
        path, load_das_umi_config(None), video_decoder_factory=FakeVideoDecoder
    )
    assert result.passed
    episode = result.episode
    assert episode is not None
    assert not episode.actions
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
