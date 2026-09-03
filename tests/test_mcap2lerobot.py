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
    nearest_value,
    resolve_fps,
    resolve_robot_type,
    source_video_codec,
)
from mcap2lerobot.reader import (
    EpisodeData,
    TimedArray,
    TimedImage,
    get_message_field,
    read_lerobot_attachments,
)


def make_episode(path: Path, offset_ns: int = 0, task_index: int | None = None) -> EpisodeData:
    timestamps = [offset_ns + value for value in (0, 33_333_333, 66_666_666)]
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
            "observation.images.head": [
                TimedImage(
                    timestamp,
                    np.full((2, 3, 3), index, dtype=np.uint8),
                )
                for index, timestamp in enumerate(timestamps)
            ]
        },
    )


class FakeReader:
    def __init__(self, episodes):
        self.episodes = episodes

    def read(self, path):
        return self.episodes[Path(path).name]


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
    assert features["observation.images.head"]["shape"] == (2, 3, 3)


def test_nearest_value_enforces_gap():
    values = [TimedArray(100, np.array([1], dtype=np.float32))]
    assert nearest_value(values, 110, 10, "action").tolist() == [1]
    with pytest.raises(ValueError, match="exceeding"):
        nearest_value(values, 111, 10, "action")


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
    assert dataset.episodes[0][1]["observation.images.head"][0, 0, 0] == 1
    assert dataset.finalized is True


EMBEDDED_INFO = {
    "codebase_version": "v3.0",
    "fps": 20,
    "robot_type": "bi_rdt_gripper",
    "features": {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["left.x", "left.y"]},
        "action": {"dtype": "float32", "shape": [1], "names": ["grip"]},
        "observation.images.head": {
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
    assert features["observation.images.head"]["shape"] == (2, 3, 3)
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
