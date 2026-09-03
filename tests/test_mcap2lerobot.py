import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mcap2lerobot.config import DEFAULT_ROBOT_TYPE, DEFAULT_TASK, ConverterConfig
from mcap2lerobot.converter import (
    McapToLeRobotConverter,
    build_features,
    discover_mcap_files,
    frame_task,
    infer_fps,
    nearest_indices,
    nearest_value,
    plan_pairing,
    resolve_fps,
    resolve_robot_type,
    source_video_codec,
)
from mcap2lerobot.reader import (
    EpisodeData,
    McapEpisodeReader,
    TimedArray,
    VideoStreamIndex,
    get_message_field,
    read_lerobot_attachments,
)

HEAD = "observation.images.head"
TIMESTAMPS = (0, 33_333_333, 66_666_666)


def make_episode(path: Path, offset_ns: int = 0, task_index: int | None = None) -> EpisodeData:
    timestamps = [offset_ns + value for value in TIMESTAMPS]
    return EpisodeData(
        source=path,
        states=[
            TimedArray(timestamp, np.array([index, index + 1], dtype=np.float32), task_index)
            for index, timestamp in enumerate(timestamps)
        ],
        actions=[
            TimedArray(timestamp, np.array([index + 2], dtype=np.float32))
            for index, timestamp in enumerate(timestamps)
        ],
        videos={
            HEAD: VideoStreamIndex(
                feature=HEAD,
                topic="observation/images/head/video",
                timestamps_ns=np.asarray(timestamps, dtype=np.int64),
                shape=(2, 3, 3),
                format="h264",
            )
        },
    )


def head_frames(count: int = len(TIMESTAMPS)) -> list[np.ndarray]:
    """Packet k of the head camera decodes to an image filled with k."""
    return [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(count)]


class FakeReader:
    """Stands in for both passes: pass one from `episodes`, pass two from `frames`."""

    def __init__(self, episodes, frames=None):
        self.episodes = episodes
        self.frames = frames or {}

    def read(self, path):
        return self.episodes[Path(path).name]

    def iter_frames(self, episode, video_index):
        per_camera = self.frames.get(episode.source.name, {HEAD: head_frames()})
        for position in range(len(episode.states)):
            yield {name: per_camera[name][int(index[position])] for name, index in video_index.items()}


class FakeDataset:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.episodes = []
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def finalize(self):
        self.finalized = True


def test_topic_to_video_feature_name():
    config = ConverterConfig()
    assert (
        config.video_feature_name("/observation/images/head/video")
        == "observation.images.head"
    )
    assert config.video_feature_name("observation/images/head/image") is None


def test_message_field_supports_namespace_and_mapping():
    assert get_message_field(SimpleNamespace(action=[1]), "action") == [1]
    assert get_message_field({"action": [2]}, "action") == [2]
    with pytest.raises(KeyError):
        get_message_field(SimpleNamespace(), "action")


def test_infers_fps_and_builds_features(tmp_path):
    episode = make_episode(tmp_path / "episode_000.mcap")
    assert infer_fps(episode) == 30
    features = build_features(episode)
    assert features["observation.state"]["shape"] == (2,)
    assert features["action"]["shape"] == (1,)
    assert features[HEAD]["shape"] == (2, 3, 3)


def test_nearest_value_enforces_gap():
    values = [TimedArray(100, np.array([1], dtype=np.float32))]
    assert nearest_value(values, 110, 10, "action").tolist() == [1]
    with pytest.raises(ValueError, match="exceeding"):
        nearest_value(values, 111, 10, "action")


def test_nearest_indices_matches_the_single_query_form():
    """The vectorised pairing must reproduce nearest_value, ties included."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        source = np.unique(rng.integers(0, 10_000, size=rng.integers(1, 40)))
        target = rng.integers(-500, 10_500, size=25)
        # Force some exact mid-points, where the tie rule decides.
        midpoints = ((source[:-1] + source[1:]) // 2)[:3]
        target[: midpoints.size] = midpoints
        values = [TimedArray(int(ts), np.array([ts])) for ts in source]
        expected = [int(nearest_value(values, int(ts), 10**9, "x")[0]) for ts in target]
        assert source[nearest_indices(source, target)].tolist() == expected


def test_plan_pairing_uses_the_source_gap_rule(tmp_path):
    episode = make_episode(tmp_path / "episode_000.mcap")
    plan = plan_pairing(episode, fps=30, max_sync_gap_frames=1.5)
    assert plan.action_index.tolist() == [0, 1, 2]
    assert plan.video_index[HEAD].tolist() == [0, 1, 2]
    episode.videos[HEAD].timestamps_ns = episode.videos[HEAD].timestamps_ns + 10**9
    with pytest.raises(ValueError, match="exceeding"):
        plan_pairing(episode, fps=30, max_sync_gap_frames=1.5)


def test_discovers_episode_files_in_numeric_order(tmp_path):
    for name in ("episode_10.mcap", "episode_2.mcap", "ignore.txt"):
        (tmp_path / name).touch()
    files = discover_mcap_files([tmp_path])
    assert [path.name for path in files] == ["episode_2.mcap", "episode_10.mcap"]


def test_converts_multiple_mcap_files_to_one_dataset(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    paths = [input_dir / "episode_000.mcap", input_dir / "episode_001.mcap"]
    for path in paths:
        path.touch()

    episodes = {
        paths[0].name: make_episode(paths[0]),
        paths[1].name: make_episode(paths[1], offset_ns=1_000_000_000),
    }
    created = []

    def dataset_factory(**kwargs):
        dataset = FakeDataset(kwargs)
        created.append(dataset)
        return dataset

    output = tmp_path / "output"
    converter = McapToLeRobotConverter(
        inputs=[input_dir],
        output_root=output,
        repo_id="local/test",
        config=ConverterConfig(task="pick", robot_type="test_robot"),
        reader=FakeReader(episodes),
        dataset_factory=dataset_factory,
    )

    assert converter.convert() == output.resolve()
    dataset = created[0]
    assert dataset.kwargs["fps"] == 30
    assert dataset.kwargs["use_videos"] is True
    assert len(dataset.episodes) == 2
    assert len(dataset.episodes[0]) == 3
    assert "timestamp" not in dataset.episodes[0][1]  # lerobot derives it; passing it is rejected
    assert dataset.episodes[0][1]["task"] == "pick"
    assert dataset.episodes[0][1]["action"].tolist() == [3.0]
    assert dataset.episodes[0][1][HEAD][0, 0, 0] == 1
    assert dataset.finalized is True


def test_out_of_sync_episode_fails_before_the_dataset_is_created(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    path = input_dir / "episode_000.mcap"
    path.touch()
    episode = make_episode(path)
    episode.actions = [TimedArray(ts + 10**9, a.value) for ts, a in zip(TIMESTAMPS, episode.actions)]
    factory_calls = []

    converter = McapToLeRobotConverter(
        inputs=[input_dir],
        output_root=tmp_path / "output",
        repo_id="local/test",
        reader=FakeReader({path.name: episode}),
        dataset_factory=lambda **kwargs: factory_calls.append(kwargs) or FakeDataset(kwargs),
    )
    with pytest.raises(ValueError, match="exceeding"):
        converter.convert()
    assert factory_calls == []


EMBEDDED_INFO = {
    "codebase_version": "v3.0",
    "fps": 20,
    "robot_type": "bi_rdt_gripper",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["left.x", "left.y"]},
        "action": {"dtype": "float32", "shape": [1], "names": ["grip"]},
        HEAD: {
            "dtype": "video",
            "shape": [2, 3, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.codec": "h264"},
        },
    },
}
EMBEDDED_EPISODE = {"episode_index": 0, "length": 3, "tasks": ["pick up the cup"], "task_index_to_task": {"0": "pick up the cup"}}


def _embedded_episode(path: Path) -> EpisodeData:
    episode = make_episode(path, task_index=0)
    # The MCAP schema hands the reader float64 for a float32 source; the
    # declaration is what restores the dtype.
    episode.states = [TimedArray(s.timestamp_ns, s.value.astype(np.float64), s.task_index) for s in episode.states]
    episode.info = EMBEDDED_INFO
    episode.episode_meta = EMBEDDED_EPISODE
    return episode


def test_embedded_metadata_restores_names_dtype_fps_robot_type_codec_and_task(tmp_path):
    episode = _embedded_episode(tmp_path / "episode_000.mcap")
    config = ConverterConfig()

    features = build_features(episode)
    assert features["observation.state"] == {"dtype": "float32", "shape": (2,), "names": ["left.x", "left.y"]}
    assert features["action"]["names"] == ["grip"]
    assert features[HEAD]["shape"] == (2, 3, 3)
    assert resolve_fps(config, episode) == 20
    assert resolve_robot_type(config, episode) == "bi_rdt_gripper"
    assert source_video_codec(episode) == "h264"
    assert frame_task(config, episode, episode.states[0]) == "pick up the cup"


def test_embedded_metadata_flows_into_the_written_frames(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    path = input_dir / "episode_000.mcap"
    path.touch()
    created = []

    def dataset_factory(**kwargs):
        dataset = FakeDataset(kwargs)
        created.append(dataset)
        return dataset

    McapToLeRobotConverter(
        inputs=[input_dir],
        output_root=tmp_path / "output",
        repo_id="local/test",
        reader=FakeReader({path.name: _embedded_episode(path)}),
        dataset_factory=dataset_factory,
    ).convert()

    dataset = created[0]
    assert dataset.kwargs["fps"] == 20
    assert dataset.kwargs["robot_type"] == "bi_rdt_gripper"
    assert dataset.kwargs["video_codec"] == "h264"
    frame = dataset.episodes[0][1]
    assert frame["observation.state"].dtype == np.float32
    assert frame["task"] == "pick up the cup"


def test_explicit_overrides_beat_embedded_metadata(tmp_path):
    episode = _embedded_episode(tmp_path / "episode_000.mcap")
    config = ConverterConfig(fps=30, task="override", robot_type="other")
    assert resolve_fps(config, episode) == 30
    assert resolve_robot_type(config, episode) == "other"
    assert frame_task(config, episode, episode.states[0]) == "override"


def test_without_embedded_metadata_the_old_defaults_apply(tmp_path):
    episode = make_episode(tmp_path / "episode_000.mcap")
    config = ConverterConfig()
    assert build_features(episode)["observation.state"]["names"] == ["state_0", "state_1"]
    assert resolve_fps(config, episode) == 30  # inferred
    assert resolve_robot_type(config, episode) == DEFAULT_ROBOT_TYPE
    assert source_video_codec(episode) is None
    assert frame_task(config, episode, episode.states[0]) == DEFAULT_TASK


def test_mismatched_embedded_declaration_is_ignored(tmp_path):
    episode = make_episode(tmp_path / "episode_000.mcap")
    episode.info = {"features": {"observation.state": {"dtype": "float32", "shape": [5], "names": list("abcde")}}}
    assert build_features(episode)["observation.state"]["names"] == ["state_0", "state_1"]


def test_reader_parses_the_attachments_lerobot2mcap_writes(tmp_path):
    mcap = pytest.importorskip("mcap")
    from mcap.reader import make_reader
    from mcap.writer import Writer

    path = tmp_path / "episode.mcap"
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        writer.add_attachment(0, 0, "meta/info.json", "text/plain", json.dumps(EMBEDDED_INFO).encode())
        writer.add_attachment(0, 0, "meta/episode.json", "text/plain", json.dumps(EMBEDDED_EPISODE).encode())
        writer.add_attachment(0, 0, "recording.log", "text/plain", b"not json")
        writer.finish()
    with path.open("rb") as stream:
        info, episode_meta = read_lerobot_attachments(make_reader(stream))
    assert info == EMBEDDED_INFO
    assert episode_meta == EMBEDDED_EPISODE
    del mcap


def test_rejects_changed_feature_layout(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    first_path = input_dir / "episode_000.mcap"
    second_path = input_dir / "episode_001.mcap"
    first_path.touch()
    second_path.touch()
    first = make_episode(first_path)
    second = make_episode(second_path)
    second.states[0] = TimedArray(0, np.zeros(3, dtype=np.float32))

    converter = McapToLeRobotConverter(
        inputs=[input_dir],
        output_root=tmp_path / "output",
        repo_id="local/test",
        reader=FakeReader({first_path.name: first, second_path.name: second}),
        dataset_factory=lambda **kwargs: FakeDataset(kwargs),
    )

    with pytest.raises(ValueError, match="Feature layout changed"):
        converter.convert()


# -- the real two-pass reader over a synthetic ROS2 MCAP -----------------------

VIDEO_MSGDEF = (
    "builtin_interfaces/Time timestamp\nstring frame_id\nuint8[] data\nstring format\n"
    "================================================================================\n"
    "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n"
)
STATE_MSGDEF = "float32[] observation_state\nint64 frame_index\nint64 episode_index\nint64 index\nint64 task_index\n"
ACTION_MSGDEF = "float32[] action\nint64 frame_index\nint64 episode_index\nint64 index\nint64 task_index\n"


class CountingFakeDecoder:
    """'Decodes' a packet whose single byte is its packet number into a 2x2x3 image."""

    calls: list[tuple[str, int]] = []

    def __init__(self, codec_name: str):
        self.codec_name = codec_name

    def decode(self, data: bytes) -> np.ndarray:
        packet = data[0]
        CountingFakeDecoder.calls.append((self.codec_name, packet))
        return np.full((2, 2, 3), packet, dtype=np.uint8)


def _write_synthetic_mcap(path: Path, camera_offsets_ns: dict[str, int], packets: int, states: int, fps: int = 30):
    from mcap_ros2.writer import Writer

    period = 1_000_000_000 // fps
    base = 1_000_000_000_000  # keep every log time positive despite negative camera offsets
    with path.open("wb") as stream:
        writer = Writer(stream)
        video = writer.register_msgdef("foxglove_msgs/msg/CompressedVideo", VIDEO_MSGDEF)
        state = writer.register_msgdef("robot_states/Data", STATE_MSGDEF)
        action = writer.register_msgdef("robot_actions/Data", ACTION_MSGDEF)
        for index in range(states):
            row = {"frame_index": index, "episode_index": 0, "index": index, "task_index": 0}
            log_time = base + index * period
            writer.write_message("robot/states/data", state, {"observation_state": [float(index), 0.5], **row}, log_time=log_time, publish_time=log_time, sequence=index)
            writer.write_message("robot/actions/data", action, {"action": [float(index) + 2], **row}, log_time=log_time + 1_000, publish_time=log_time, sequence=index)
        for camera, offset in camera_offsets_ns.items():
            for packet in range(packets):
                log_time = base + packet * period + offset
                writer.write_message(
                    f"observation/images/{camera}/video", video,
                    {"timestamp": {"sec": log_time // 1_000_000_000, "nanosec": log_time % 1_000_000_000}, "frame_id": camera, "data": bytes([packet]), "format": "fake"},
                    log_time=log_time, publish_time=log_time, sequence=packet,
                )
        writer.finish()


def test_two_pass_reader_decodes_only_the_packets_the_plan_needs(tmp_path):
    pytest.importorskip("mcap_ros2")
    path = tmp_path / "episode_000.mcap"
    # Head runs 5 ms early, wrist 12 ms late; both still inside 1.5 frames at 30 fps.
    _write_synthetic_mcap(path, {"head": -5_000_000, "wrist": 12_000_000}, packets=12, states=10)
    CountingFakeDecoder.calls = []
    reader = McapEpisodeReader(ConverterConfig(), video_decoder_factory=CountingFakeDecoder)

    episode = reader.read(path)
    assert len(episode.states) == 10 and len(episode.actions) == 10
    assert sorted(episode.videos) == ["observation.images.head", "observation.images.wrist"]
    assert len(episode.videos["observation.images.head"]) == 12
    assert episode.videos["observation.images.head"].shape == (2, 2, 3)
    assert episode.states[3].task_index == 0
    # Pass one decoded exactly one packet per camera, for the shape.
    assert sorted(CountingFakeDecoder.calls) == [("fake", 0), ("fake", 0)]

    plan = plan_pairing(episode, fps=30, max_sync_gap_frames=1.5)
    assert plan.video_index["observation.images.head"].tolist() == list(range(10))
    assert plan.action_index.tolist() == list(range(10))

    CountingFakeDecoder.calls = []
    frames = list(reader.iter_frames(episode, plan.video_index))
    assert len(frames) == 10
    for position, images in enumerate(frames):
        assert images["observation.images.head"][0, 0, 0] == plan.video_index["observation.images.head"][position]
        assert images["observation.images.wrist"][0, 0, 0] == plan.video_index["observation.images.wrist"][position]
    # Pass two decoded each needed packet once; packets 10 and 11 were never touched.
    assert len(CountingFakeDecoder.calls) == 20
    assert max(packet for _, packet in CountingFakeDecoder.calls) == 9


def test_two_pass_reader_shares_a_packet_between_states_without_redecoding(tmp_path):
    pytest.importorskip("mcap_ros2")
    path = tmp_path / "episode_000.mcap"
    _write_synthetic_mcap(path, {"head": 0}, packets=10, states=10)
    reader = McapEpisodeReader(ConverterConfig(), video_decoder_factory=CountingFakeDecoder)
    episode = reader.read(path)
    # Two neighbouring states pointed at the same packet, as a stalled camera would do.
    index = np.array([0, 1, 2, 3, 3, 5, 6, 7, 8, 9])
    CountingFakeDecoder.calls = []
    frames = list(reader.iter_frames(episode, {"observation.images.head": index}))
    assert [f["observation.images.head"][0, 0, 0] for f in frames] == index.tolist()
    assert len(CountingFakeDecoder.calls) == 9


def test_two_pass_reader_reports_a_camera_that_runs_out(tmp_path):
    pytest.importorskip("mcap_ros2")
    path = tmp_path / "episode_000.mcap"
    _write_synthetic_mcap(path, {"head": 0}, packets=10, states=10)
    reader = McapEpisodeReader(ConverterConfig(), video_decoder_factory=CountingFakeDecoder)
    episode = reader.read(path)
    with pytest.raises(ValueError, match="ran out"):
        list(reader.iter_frames(episode, {"observation.images.head": np.arange(10) + 3}))
