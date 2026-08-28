"""Unit tests for the HDF5 -> LeRobot v3.0 converter.

Everything runs on a synthetic two-episode recording built in a temp directory,
so the suite needs no real data on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hdf52lerobot.config import ConverterConfig  # noqa: E402
from hdf52lerobot.converter import (  # noqa: E402
    Hdf5ToLeRobotConverter,
    build_action,
    discover_episodes,
)
from hdf52lerobot.geometry import (  # noqa: E402
    quaternions_to_rotation_6d,
    rotation_6d_to_quaternions,
    to_wxyz,
)
from hdf52lerobot.source import read_episode  # noqa: E402
from hdf52lerobot.stats import estimate_num_samples  # noqa: E402
from hdf52lerobot.sync import build_sync, nearest_indices  # noqa: E402

FPS = 20
FRAMES = 12
WIDTH, HEIGHT = 64, 48
GRIPPER_LO, GRIPPER_HI = 100.0, 200.0


def _write_hdf5(path: Path, values: np.ndarray, timestamps: np.ndarray) -> None:
    import h5py

    with h5py.File(path, "w") as handle:
        handle.create_dataset("float_array", data=values)
        handle.create_dataset("timestamp_array", data=timestamps)


def _write_video(path: Path, frames: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", FPS, options={"crf": "30", "preset": "ultrafast"})
        stream.pix_fmt = "yuv420p"
        stream.width, stream.height = WIDTH, HEIGHT
        for index in range(frames):
            image = np.full((HEIGHT, WIDTH, 3), (index * 7) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _make_episode(directory: Path, frames: int = FRAMES, jitter: float = 0.004) -> Path:
    """Write one synthetic recording directory with the full stream set."""
    directory.mkdir(parents=True)
    base = 1_700_000_000.0 + np.arange(frames) / FPS
    rng = np.random.default_rng(0)

    for side, tracker in (("left", "LHR-AAAA"), ("right", "LHR-BBBB")):
        angles = np.linspace(0.0, 0.6, frames)
        # Same column order as the rig: [qx, qy, qz, qw].
        quaternions = np.stack(
            [np.sin(angles / 2), np.zeros(frames), np.zeros(frames), np.cos(angles / 2)], axis=1
        )
        positions = np.stack([angles, -angles, np.full(frames, 0.5)], axis=1)
        _write_hdf5(
            directory / f"{side}_umi_036X_tracker_{tracker}.hdf5",
            np.concatenate([positions, quaternions], axis=1),
            base + rng.uniform(-jitter, jitter, frames),
        )
        _write_hdf5(
            directory / f"{side}_umi_036X_openv2_{GRIPPER_LO}_{GRIPPER_HI}.hdf5",
            np.linspace(GRIPPER_HI, GRIPPER_LO, frames).reshape(-1, 1),
            base + rng.uniform(-jitter, jitter, frames),
        )

    videos = [
        "head_036H_mvs_mono_DA1_libx264.mkv",
        "left_umi_036X_mvs_mono_DA2_libx264.mkv",
        "right_umi_036X_mvs_mono_DA3_libx264.mkv",
        "left_umi_036X_tactile_left_OG1_ffv1.mkv",
        "left_umi_036X_tactile_right_OG2_ffv1.mkv",
        "right_umi_036X_tactile_left_OG3_ffv1.mkv",
        "right_umi_036X_tactile_right_OG4_ffv1.mkv",
    ]
    for name in videos:
        _write_video(directory / name, frames)
        _write_hdf5(
            directory / f"{name}.timestamp.hdf5",
            np.zeros((frames, 1, 1)),
            base + rng.uniform(-jitter, jitter, frames),
        )
    return directory


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("recording")
    _make_episode(root / "ep_0001")
    _make_episode(root / "ep_0002")
    return root


@pytest.fixture(scope="module")
def dataset(recording: Path, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("dataset") / "out"
    config = ConverterConfig(fps=FPS, task="unit test", gop=4, preset="ultrafast")
    return Hdf5ToLeRobotConverter([recording], output, config).convert()


# -- geometry ----------------------------------------------------------------


def test_rotation_6d_round_trip():
    quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 1.0, 0.0]])
    recovered = rotation_6d_to_quaternions(quaternions_to_rotation_6d(quaternions))
    assert np.allclose(recovered, quaternions, atol=1e-9)


def test_rotation_6d_is_two_orthonormal_columns():
    rotations = quaternions_to_rotation_6d(np.array([[0.5, 0.5, 0.5, 0.5]]))
    first, second = rotations[0, :3], rotations[0, 3:]
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.isclose(np.linalg.norm(second), 1.0)
    assert np.isclose(first @ second, 0.0, atol=1e-12)


def test_quaternion_order_swap():
    assert np.allclose(to_wxyz(np.array([[1.0, 2.0, 3.0, 4.0]]), "xyzw"), [[4.0, 1.0, 2.0, 3.0]])
    assert np.allclose(to_wxyz(np.array([[1.0, 2.0, 3.0, 4.0]]), "wxyz"), [[1.0, 2.0, 3.0, 4.0]])


# -- source / sync -----------------------------------------------------------


def test_read_episode_classifies_every_stream(recording: Path):
    episode = read_episode(recording / "ep_0001", ConverterConfig(fps=FPS))
    assert sorted(episode.poses) == ["left_tcp", "right_tcp"]
    assert sorted(episode.grippers) == ["left_gripper", "right_gripper"]
    assert sorted(episode.videos) == [
        "head",
        "left_tactile_left",
        "left_tactile_right",
        "left_wrist",
        "right_tactile_left",
        "right_tactile_right",
        "right_wrist",
    ]
    assert episode.grippers["left_gripper"].lo == GRIPPER_LO
    assert episode.grippers["left_gripper"].hi == GRIPPER_HI


def test_gripper_normalizes_open_to_one(recording: Path):
    episode = read_episode(recording / "ep_0001", ConverterConfig(fps=FPS))
    normalized = episode.grippers["left_gripper"].normalized()
    assert np.isclose(normalized[0], 1.0)
    assert np.isclose(normalized[-1], 0.0)
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0


def test_index_pairing_is_identity_when_lengths_match(recording: Path):
    config = ConverterConfig(fps=FPS)
    report = build_sync(read_episode(recording / "ep_0001", config), config)
    for stream in report.streams.values():
        assert stream.mode == "index"
        assert stream.replaced == 0


def test_nearest_pairing_picks_the_closest_sample():
    source = np.array([0.0, 0.1, 0.2])
    master = np.array([0.04, 0.11, 0.5])
    assert nearest_indices(source, master).tolist() == [0, 1, 2]


def test_sync_rejects_a_stream_that_drifts_too_far(tmp_path: Path):
    directory = _make_episode(tmp_path / "drifted")
    stream = directory / "left_umi_036X_openv2_100.0_200.0.hdf5"
    import h5py

    with h5py.File(stream, "r+") as handle:
        handle["timestamp_array"][...] = handle["timestamp_array"][...] + 1.0
    config = ConverterConfig(fps=FPS)
    with pytest.raises(ValueError, match="not synchronized"):
        build_sync(read_episode(directory, config), config)


def test_missing_state_stream_is_rejected(tmp_path: Path):
    directory = _make_episode(tmp_path / "incomplete")
    (directory / "right_umi_036X_tracker_LHR-BBBB.hdf5").unlink()
    with pytest.raises(ValueError, match="right_tcp"):
        read_episode(directory, ConverterConfig(fps=FPS))


# -- actions -----------------------------------------------------------------


def test_action_is_the_next_state():
    states = np.arange(12, dtype=np.float32).reshape(4, 3)
    actions = build_action(states)
    assert np.array_equal(actions[:-1], states[1:])
    assert np.array_equal(actions[-1], states[-1])


def test_estimate_num_samples_matches_lerobot():
    assert estimate_num_samples(10) == 10
    assert estimate_num_samples(1000) == 177
    assert estimate_num_samples(20_000) == 1681


def test_discover_episodes_accepts_parent_or_episode(recording: Path):
    assert discover_episodes([recording]) == [recording / "ep_0001", recording / "ep_0002"]
    assert discover_episodes([recording / "ep_0002"]) == [recording / "ep_0002"]


# -- end to end --------------------------------------------------------------


def test_dataset_tree(dataset: Path):
    assert (dataset / "meta" / "info.json").is_file()
    assert (dataset / "meta" / "stats.json").is_file()
    assert (dataset / "meta" / "tasks.parquet").is_file()
    assert (dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet").is_file()
    assert (dataset / "data" / "chunk-000" / "file-000.parquet").is_file()
    for key in ("head", "left_wrist", "right_tactile_right"):
        assert (
            dataset / "videos" / f"observation.images.{key}" / "chunk-000" / "file-000.mp4"
        ).is_file()


def test_info_matches_the_taccap_layout(dataset: Path):
    info = json.loads((dataset / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 2 * FRAMES
    assert info["splits"] == {"train": "0:2"}
    assert info["features"]["observation.state"]["shape"] == [20]
    assert info["features"]["observation.state"]["names"][:3] == [
        "left_tcp.x",
        "left_tcp.y",
        "left_tcp.z",
    ]
    head = info["features"]["observation.images.head"]
    assert head["dtype"] == "video"
    assert head["shape"] == [HEIGHT, WIDTH, 3]
    assert head["info"]["video.codec"] == "h264"
    assert head["info"]["video.fps"] == FPS


def test_data_parquet_schema_and_values(dataset: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet")
    assert table.column_names == [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    import pyarrow as pa

    state_type = table.schema.field("observation.state").type
    assert pa.types.is_fixed_size_list(state_type)
    assert state_type.list_size == 20
    assert state_type.value_type == pa.float32()

    frame = table.to_pandas()
    assert frame["index"].tolist() == list(range(2 * FRAMES))
    assert frame["frame_index"].tolist() == list(range(FRAMES)) * 2
    assert np.allclose(frame["timestamp"].to_numpy()[:FRAMES], np.arange(FRAMES) / FPS)

    states = np.stack(frame["observation.state"].to_numpy())
    actions = np.stack(frame["action"].to_numpy())
    first = states[:FRAMES]
    assert np.allclose(actions[: FRAMES - 1], first[1:])
    assert np.isclose(first[0, 9], 1.0)  # gripper starts open


def test_episodes_parquet_locates_every_video(dataset: Path):
    import pyarrow.parquet as pq

    frame = pq.read_table(
        dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()
    assert frame["episode_index"].tolist() == [0, 1]
    assert frame["length"].tolist() == [FRAMES, FRAMES]
    assert frame["dataset_from_index"].tolist() == [0, FRAMES]
    assert frame["dataset_to_index"].tolist() == [FRAMES, 2 * FRAMES]
    prefix = "videos/observation.images.head"
    assert frame[f"{prefix}/from_timestamp"].tolist() == [0.0, FRAMES / FPS]
    assert frame[f"{prefix}/to_timestamp"].tolist() == [FRAMES / FPS, 2 * FRAMES / FPS]


def test_videos_are_constant_frame_rate(dataset: Path):
    import av

    path = dataset / "videos" / "observation.images.head" / "chunk-000" / "file-000.mp4"
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        timestamps = [float(frame.pts * stream.time_base) for frame in container.decode(video=0)]
    assert len(timestamps) == 2 * FRAMES
    expected = np.arange(2 * FRAMES) / FPS
    # LeRobot's default tolerance_s is 1e-4; frames must sit exactly on i/fps.
    assert np.abs(np.array(timestamps) - expected).max() < 1e-4


def test_stats_cover_every_feature(dataset: Path):
    info = json.loads((dataset / "meta" / "info.json").read_text())
    stats = json.loads((dataset / "meta" / "stats.json").read_text())
    assert sorted(stats) == sorted(info["features"])
    assert sorted(stats["observation.state"]) == ["count", "max", "mean", "min", "std"]
    assert stats["observation.state"]["count"] == [2 * FRAMES]
    assert np.array(stats["observation.images.head"]["mean"]).shape == (3, 1, 1)
    assert 0.0 <= np.array(stats["observation.images.head"]["mean"]).min() <= 1.0


def test_refuses_to_overwrite_by_default(recording: Path, dataset: Path):
    with pytest.raises(FileExistsError):
        Hdf5ToLeRobotConverter([recording], dataset, ConverterConfig(fps=FPS)).convert()


def test_shipped_config_loads():
    config = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "hdf52lerobot.yaml"
    )
    assert config.fps == 20
    assert config.pair == "index"
    assert config.quat_order == "xyzw"
    assert config.robot_type == "bi_rtc_gripper"
    assert config.master_clock == "left_tcp"
    assert config.state_layout == ("left_tcp", "left_gripper", "right_tcp", "right_gripper")
    assert config.warn_sync_gap_frames < config.max_sync_gap_frames
    kinds = {"left_tcp": "pose", "left_gripper": "gripper", "right_tcp": "pose", "right_gripper": "gripper"}
    assert len(config.state_names(kinds)) == 20


def test_nearest_pairing_converts_too(recording: Path, tmp_path: Path):
    config = ConverterConfig(fps=FPS, pair="nearest", gop=4, preset="ultrafast")
    output = Hdf5ToLeRobotConverter([recording], tmp_path / "nearest", config).convert()
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_frames"] == 2 * FRAMES


def test_cli_round_trip(recording: Path, tmp_path: Path):
    from hdf52lerobot.cli import main

    output = tmp_path / "cli"
    code = main(
        [
            "--input", str(recording),
            "--output", str(output),
            "--task", "cli test",
            "--fps", str(FPS),
            "--gop", "4",
            "--preset", "ultrafast",
        ]
    )
    assert code == 0
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["fps"] == FPS
    assert info["total_episodes"] == 2

    import pyarrow.parquet as pq

    tasks = pq.read_table(output / "meta" / "tasks.parquet").to_pandas()
    assert tasks.index.name == "task"
    assert tasks.index.tolist() == ["cli test"]


def test_cli_reports_failure_without_raising(tmp_path: Path):
    from hdf52lerobot.cli import main

    empty = tmp_path / "empty"
    (empty / "nothing").mkdir(parents=True)
    assert main(["--input", str(empty), "--output", str(tmp_path / "out"), "--task", "x"]) == 1
