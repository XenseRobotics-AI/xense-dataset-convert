from __future__ import annotations

import numpy as np

from rlds2lerobot.config import ConverterConfig
from rlds2lerobot.converter import RldsToLeRobotConverter, build_features
from xense_dataset_convert._lerobot_writer import writer_parallelism_kwargs


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


def test_missing_configured_image_is_allowed():
    data = episode()
    del data["steps"]["observation"]["exterior_image_2_left"]
    config = ConverterConfig(image_fields=("exterior_image_1_left", "missing"))
    features = build_features(data, config)
    assert "observation.images.exterior_image_1_left" in features
    assert "observation.images.missing" not in features


def test_video_encoder_threads_can_be_overridden(monkeypatch):
    class Dataset:
        @classmethod
        def create(cls, image_writer_threads=0, encoder_threads=None):
            pass

    monkeypatch.setattr("xense_dataset_convert._lerobot_writer.available_cpu_count", lambda: 16)
    features = {
        "observation.images.left": {"dtype": "video"},
        "observation.images.right": {"dtype": "video"},
    }

    assert writer_parallelism_kwargs(Dataset, features, 3, None) == {
        "image_writer_threads": 3,
        "encoder_threads": 8,
    }
    assert writer_parallelism_kwargs(Dataset, features, 3, 2)["encoder_threads"] == 2


def test_parallelism_config_rejects_invalid_thread_counts():
    import pytest

    with pytest.raises(ValueError, match="image_writer_threads"):
        ConverterConfig(image_writer_threads=-1)
    with pytest.raises(ValueError, match="encoder_threads"):
        ConverterConfig(encoder_threads=0)
