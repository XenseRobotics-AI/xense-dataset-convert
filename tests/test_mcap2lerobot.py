from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mcap2lerobot.config import ConverterConfig
from mcap2lerobot.converter import (
    McapToLeRobotConverter,
    build_features,
    discover_mcap_files,
    infer_fps,
    nearest_value,
)
from mcap2lerobot.reader import EpisodeData, TimedArray, TimedImage, get_message_field


def make_episode(path: Path, offset_ns: int = 0) -> EpisodeData:
    timestamps = [offset_ns + value for value in (0, 33_333_333, 66_666_666)]
    return EpisodeData(
        source=path,
        states=[
            TimedArray(timestamp, np.array([index, index + 1], dtype=np.float32))
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
    assert dataset.episodes[0][1]["timestamp"] == pytest.approx(1 / 30)
    assert dataset.episodes[0][1]["task"] == "pick"
    assert dataset.episodes[0][1]["action"].tolist() == [3.0]
    assert dataset.episodes[0][1]["observation.images.head"][0, 0, 0] == 1
    assert dataset.finalized is True


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
