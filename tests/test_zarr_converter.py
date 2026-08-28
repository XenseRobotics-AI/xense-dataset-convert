import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from lerobot2zarr.zarr_converter import LeRobotZarrConverter


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_converts_dynamic_features_and_episode_boundaries(tmp_path):
    rows = []
    for index, episode_index in enumerate([4, 4, 9]):
        rows.append(
            {
                "action": np.array([index, index + 1], dtype=np.float32),
                "observation.state": np.array([index], dtype=np.float64),
                "observation.images.head_left": np.full(
                    (3, 4, 5), index / 2, dtype=np.float32
                ),
                "observation.images.head_right": np.full(
                    (2, 4, 3), index, dtype=np.uint8
                ),
                "episode_index": np.int64(episode_index),
                "task": "pick" if episode_index == 4 else "place",
            }
        )

    output = tmp_path / "dataset.zarr"
    converter = LeRobotZarrConverter(
        FakeDataset(rows),
        output,
        feature_metadata={
            "action": {"dtype": "float32"},
            "observation.state": {"dtype": "float64"},
            "observation.images.head_left": {"dtype": "video"},
            "observation.images.head_right": {"dtype": "video"},
            "episode_index": {"dtype": "int64"},
        },
    )

    converter.convert()

    root = zarr.open_group(output, mode="r")
    assert root["data/action"].shape == (3, 2)
    assert root["data/observation/state"].shape == (3, 1)
    assert root["data/observation/images/head_left"].shape == (3, 4, 5, 3)
    assert root["data/observation/images/head_right"].shape == (3, 2, 4, 3)
    assert root["data/observation/images/head_left"].dtype == np.dtype("uint8")
    assert root["meta/episode_ends"][:].tolist() == [2, 3]
    assert root["meta/episode_ids"][:].tolist() == [4, 9]

    schema = json.loads((output / "meta" / "schema.json").read_text())
    assert schema["frame_count"] == 3
    assert schema["episode_count"] == 2
    assert schema["features"]["observation.images.head_left"]["kind"] == "video"
    assert schema["features"]["task"]["dtype"] == "<U5"


def test_rejects_feature_shape_changes(tmp_path):
    dataset = FakeDataset(
        [
            {"action": np.zeros(2, dtype=np.float32), "episode_index": 0},
            {"action": np.zeros(3, dtype=np.float32), "episode_index": 0},
        ]
    )

    with pytest.raises(ValueError, match="changed shape"):
        LeRobotZarrConverter(dataset, tmp_path / "bad.zarr").convert()


def test_rejects_noncontiguous_episode_frames(tmp_path):
    dataset = FakeDataset(
        [
            {"action": np.zeros(1), "episode_index": episode_index}
            for episode_index in [0, 1, 0]
        ]
    )

    with pytest.raises(ValueError, match="not contiguous"):
        LeRobotZarrConverter(dataset, tmp_path / "bad.zarr").convert()
