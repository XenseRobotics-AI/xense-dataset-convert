from __future__ import annotations

import numpy as np

from rlds2lerobot.config import ConverterConfig
from rlds2lerobot.converter import RldsToLeRobotConverter, build_features


def episode(length: int = 2) -> dict:
    return {
        "episode_metadata": {"file_path": b"demo"},
        "steps": {
            "action": np.ones((length, 7), dtype=np.float64),
            "language_instruction": np.asarray([b"insert earbuds"] * length),
            "observation": {
                "joint_position": np.ones((length, 7), dtype=np.float64),
                "gripper_position": np.ones((length, 1), dtype=np.float64),
                "exterior_image_1_left": np.zeros((length, 2, 3, 3), dtype=np.uint8),
                "exterior_image_2_left": np.zeros((length, 2, 3, 3), dtype=np.uint8),
            },
        },
    }


class FakeDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.saved = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        self.finalized = True


def test_droid_features_and_streaming_conversion(tmp_path):
    first = episode()
    second = episode(1)
    created = {}

    def factory(**kwargs):
        created["dataset"] = FakeDataset(**kwargs)
        return created["dataset"]

    converter = RldsToLeRobotConverter(
        tmp_path / "source",
        tmp_path / "output",
        ConverterConfig(max_episodes=2),
        episode_reader=lambda *_: iter((first, second)),
        dataset_factory=factory,
    )
    result = converter.convert()

    dataset = created["dataset"]
    assert result.name == "output"
    assert len(dataset.frames) == 3
    assert dataset.saved == 2
    assert dataset.finalized
    assert dataset.frames[0]["observation.state"].shape == (8,)
    assert dataset.frames[0]["action"].shape == (7,)
    assert dataset.frames[0]["task"] == "insert earbuds"
    assert dataset.kwargs["use_videos"] is True


def test_removes_empty_image_dirs_after_successful_finalize(tmp_path):
    output = tmp_path / "output"
    retained_file = output / "images" / "retained" / "frame.png"

    class DatasetWithImageDirs(FakeDataset):
        def finalize(self):
            empty_dir = self.kwargs["root"] / "images" / "observation.images.head"
            empty_dir.mkdir(parents=True)
            retained_file.parent.mkdir(parents=True)
            retained_file.touch()
            super().finalize()

    converter = RldsToLeRobotConverter(
        tmp_path / "source",
        output,
        ConverterConfig(max_episodes=1, show_progress=False),
        episode_reader=lambda *_: iter((episode(),)),
        dataset_factory=lambda **kwargs: DatasetWithImageDirs(**kwargs),
    )

    converter.convert()

    assert not (output / "images" / "observation.images.head").exists()
    assert retained_file.is_file()
    assert (output / "images" / "retained").is_dir()


def test_missing_configured_image_is_allowed():
    data = episode()
    del data["steps"]["observation"]["exterior_image_2_left"]
    config = ConverterConfig(image_fields=("exterior_image_1_left", "missing"))
    features = build_features(data, config)
    assert "observation.images.exterior_image_1_left" in features
    assert "observation.images.missing" not in features
