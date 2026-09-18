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

from dataclasses import replace  # noqa: E402

from hdf52lerobot.config import ConverterConfig, StreamRule  # noqa: E402
from hdf52lerobot.converter import (  # noqa: E402
    Hdf5ToLeRobotConverter,
    build_action,
    discover_episodes,
)
from hdf52lerobot.imaging import (  # noqa: E402
    bgr_to_rgb,
    compose,
    transpose,
    transformed_size,
    validate_transforms,
)
from hdf52lerobot.geometry import (  # noqa: E402
    quaternions_to_rotation_6d,
    rotation_6d_to_quaternions,
    to_wxyz,
)
from hdf52lerobot.source import read_episode  # noqa: E402
from hdf52lerobot.stats import StatsAccumulator, estimate_num_samples  # noqa: E402
from hdf52lerobot.source import GripperStream, PoseStream, SourceEpisode, VideoStream  # noqa: E402
from hdf52lerobot.sync import build_sync, nearest_indices, uniform_grid  # noqa: E402

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


# -- frame transforms --------------------------------------------------------


def test_bgr_to_rgb_and_transpose_match_the_capture_path():
    """Same two steps `xense-taccap-lerobot` applies to a tactile frame."""
    frame = np.zeros((7, 4, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)  # BGR as the SDK hands it over
    frame[6, 3] = (40, 50, 60)

    swapped = compose(["bgr_to_rgb"])(frame)
    assert swapped[0, 0].tolist() == [30, 20, 10]
    assert swapped.shape == frame.shape

    laid_down = compose(["transpose"])(frame)
    assert laid_down.shape == (4, 7, 3)
    assert laid_down[0, 0].tolist() == [10, 20, 30]
    assert laid_down[3, 6].tolist() == [40, 50, 60]  # the corner moved with it

    both = compose(["bgr_to_rgb", "transpose"])(frame)
    assert both.shape == (4, 7, 3)
    assert both[0, 0].tolist() == [30, 20, 10]
    assert both.flags["C_CONTIGUOUS"]
    # Composing must give exactly what applying the steps one at a time gives,
    # even though it copies once instead of per step.
    assert np.array_equal(both, transpose(bgr_to_rgb(frame)))


def test_transformed_size_swaps_only_for_transpose():
    assert transformed_size(400, 700, []) == (400, 700)
    assert transformed_size(400, 700, ["bgr_to_rgb"]) == (400, 700)
    assert transformed_size(400, 700, ["bgr_to_rgb", "transpose"]) == (700, 400)


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="Unknown frame transform"):
        validate_transforms(["rotate90"])
    with pytest.raises(ValueError, match="Unknown frame transform"):
        ConverterConfig(streams=(StreamRule(r"^x\.mkv$", "video", "x", ("flip",)),))
    with pytest.raises(ValueError, match="only meaningful for 'video'"):
        StreamRule(r"^x\.hdf5$", "pose", "x", ("transpose",))


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


def _stalling_episode(tmp_path: Path) -> SourceEpisode:
    """A recorder loop at 30 Hz that stalls twice, every stream stamped a few ms later."""
    base = 1_700_000_000.0
    intervals = [0.033] * 6 + [0.150] + [0.033] * 5 + [0.100] + [0.033] * 6
    master = base + np.concatenate([[0.0], np.cumsum(intervals)])
    n = len(master)
    poses = np.zeros((n, 7)); poses[:, 6] = 1.0; poses[:, 0] = np.arange(n)  # x counts iterations
    episode = SourceEpisode(path=tmp_path / "stalling")
    for side, offset in (("left", 0.0), ("right", 0.004)):
        episode.poses[f"{side}_tcp"] = PoseStream(f"{side}_tcp", tmp_path / f"{side}.hdf5", master + offset, poses.copy())
        episode.grippers[f"{side}_gripper"] = GripperStream(
            f"{side}_gripper", tmp_path / f"{side}_g.hdf5", master + offset + 0.002, np.linspace(0, 100, n), 0.0, 100.0
        )
    episode.videos["head"] = VideoStream("head", tmp_path / "head.mkv", master + 0.006, 64, 48)
    return episode


def test_grid_mode_resamples_iterations_onto_a_uniform_20hz_grid(tmp_path: Path):
    episode = _stalling_episode(tmp_path)
    config = ConverterConfig(fps=20, pair="grid", warn_sync_gap_frames=2.0, max_sync_gap_frames=3.0)
    report = build_sync(episode, config)

    master = episode.poses["left_tcp"].timestamps
    assert report.mode == "grid"
    assert report.length == int(np.floor((master[-1] - master[0]) * 20)) + 1
    assert np.allclose(uniform_grid(master, 20), master[0] + np.arange(report.length) / 20)
    # Every tick took the iteration nearest to it; the stalls show up as repeats.
    ticks = master[0] + np.arange(report.length) / 20
    expected = np.array([int(np.argmin(np.abs(master - t))) for t in ticks])
    assert report.indices_for("left_tcp").tolist() == expected.tolist()
    assert report.duplicated >= 1 and report.dropped >= 1
    assert report.tick_timestamps.tolist() == master[expected].tolist()
    assert np.abs(report.warp_s).max() <= 0.075 + 1e-9
    # Streams still pair by iteration, so all of them follow the same index list.
    for key in ("right_tcp", "left_gripper", "head"):
        assert report.indices_for(key).tolist() == expected.tolist()
    assert "duplicated=" in report.describe()[0] and "grid 20Hz" in report.describe()[-1]


def test_grid_mode_rejects_a_stall_longer_than_the_limit(tmp_path: Path):
    episode = _stalling_episode(tmp_path)
    master = episode.poses["left_tcp"].timestamps
    master[7:] += 0.4  # one 550 ms hole in the master clock
    for stream in episode.all_streams().values():
        stream.timestamps[7:] += 0.0 if stream.key == "left_tcp" else 0.4
    with pytest.raises(ValueError, match="not synchronized"):
        build_sync(episode, ConverterConfig(fps=20, pair="grid", max_sync_gap_frames=3.0))


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


def test_image_stats_match_a_plain_numpy_reduction():
    """Running moments over batches must equal reducing every pixel at once."""
    from hdf52lerobot.stats import _prepare_image, reduce_images

    rng = np.random.default_rng(1)
    # More frames than one batch, so the batching itself is exercised.
    frames = [rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(150)]
    prepared = [_prepare_image(frame) for frame in frames]
    pixels = np.concatenate([frame.reshape(3, -1) for frame in prepared], axis=1) / 255.0
    expected = {
        "min": pixels.min(axis=1),
        "max": pixels.max(axis=1),
        "mean": pixels.mean(axis=1),
        "std": pixels.std(axis=1),
    }

    accumulator = StatsAccumulator()
    accumulator.add_episode({}, {"cam": frames[:70]})
    accumulator.add_episode({}, {"cam": frames[70:]})
    streamed = accumulator.finalize()["cam"]
    direct = reduce_images(prepared)

    for stat, want in expected.items():
        assert np.allclose(np.array(streamed[stat]).ravel(), want, atol=1e-12), stat
        assert np.allclose(np.array(direct[stat]).ravel(), want, atol=1e-12), stat
    assert streamed["count"] == direct["count"] == [150]


def test_downsample_hwc_returns_a_contiguous_copy_at_lerobot_size():
    from hdf52lerobot.stats import _prepare_image, downsample_hwc

    frame = np.zeros((700, 400, 3), dtype=np.uint8)
    small = downsample_hwc(frame)
    assert small.shape == (175, 100, 3) and small.flags["C_CONTIGUOUS"] and small.base is None
    assert _prepare_image(small).shape == (3, 175, 100)  # a no-op resize afterwards


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


def test_info_key_order_matches_tacverse(dataset: Path):
    """`features` is hundreds of lines; the totals a reader wants come first."""
    info = json.loads((dataset / "meta" / "info.json").read_text())
    assert list(info) == [
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "chunks_size",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "fps",
        "splits",
        "data_path",
        "video_path",
        "features",
    ]


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


def test_episodes_parquet_carries_per_episode_stats(dataset: Path):
    """LeRobot's dataset tools re-aggregate ``stats/*`` when episodes are edited."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads((dataset / "meta" / "info.json").read_text())
    for feature in info["features"]:
        for stat in ("min", "max", "mean", "std", "count"):
            assert f"stats/{feature}/{stat}" in table.column_names, (feature, stat)

    assert table.schema.field("stats/observation.state/count").type == pa.list_(pa.int64())
    assert table.schema.field("stats/observation.state/mean").type == pa.list_(pa.float64())
    assert table.schema.field("stats/observation.images.head/mean").type == pa.list_(
        pa.list_(pa.list_(pa.float64()))
    )

    rows = table.to_pylist()
    assert [row["stats/observation.state/count"] for row in rows] == [[FRAMES], [FRAMES]]
    assert len(rows[0]["stats/observation.state/mean"]) == 20
    assert np.array(rows[0]["stats/observation.images.head/mean"]).shape == (3, 1, 1)
    # Frame 0 of episode 1 has global index FRAMES, so its per-episode minimum says so.
    assert rows[1]["stats/index/min"] == [float(FRAMES)]

    # Per-episode blocks re-aggregate to the dataset-wide stats.json.
    stats = json.loads((dataset / "meta" / "stats.json").read_text())
    means = np.array([row["stats/observation.state/mean"] for row in rows])
    counts = np.array([row["stats/observation.state/count"] for row in rows])
    assert np.allclose((means * counts).sum(axis=0) / counts.sum(), stats["observation.state"]["mean"])


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


def test_tacverse_config_loads():
    config = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "tacverse_rdt.yaml"
    )
    assert (config.fps, config.pair, config.keep_source_timestamp) == (20, "grid", True)
    # Encoding runs on the GPU; the dataset still records plain h264.
    assert config.vcodec == "h264_nvenc" and config.codec_threads == 1
    assert config.encoder_options()["rc"] == "constqp"
    assert config.robot_type == "bi_rdt_gripper"
    # The third camera is a side view without a pose, and the tactile streams
    # carry the two transforms the capture path would have applied.
    by_key = {rule.key: rule for rule in config.streams}
    assert "head" not in by_key
    assert by_key["side"].kind == "video" and by_key["side"].transform == ()
    assert by_key["{side}_tactile_{pos}"].transform == ("bgr_to_rgb", "transpose")
    assert len(config.state_names({"left_tcp": "pose", "left_gripper": "gripper", "right_tcp": "pose", "right_gripper": "gripper"})) == 20


def test_transformed_camera_is_written_rotated_and_recoloured(recording: Path, tmp_path: Path):
    """A camera with `transform` lands in the dataset the other way round."""
    import av

    base = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "hdf52lerobot.yaml"
    )
    base = replace(base, fps=FPS, gop=4, preset="ultrafast", task="t")
    with_transform = replace(
        base,
        streams=tuple(
            replace(rule, transform=("bgr_to_rgb", "transpose"))
            if "tactile" in rule.key
            else rule
            for rule in base.streams
        ),
    )

    def first_frame(root: Path, key: str) -> np.ndarray:
        path = root / "videos" / f"observation.images.{key}" / "chunk-000" / "file-000.mp4"
        with av.open(str(path)) as container:
            return next(container.decode(video=0)).to_ndarray(format="rgb24")

    plain = Hdf5ToLeRobotConverter([recording], tmp_path / "plain", base).convert()
    turned = Hdf5ToLeRobotConverter(
        [recording], tmp_path / "turned", with_transform
    ).convert()

    info = json.loads((turned / "meta" / "info.json").read_text())
    # Sources are WIDTH x HEIGHT; only the transposed cameras come out swapped.
    assert info["features"]["observation.images.head"]["shape"] == [HEIGHT, WIDTH, 3]
    tactile = info["features"]["observation.images.left_tactile_left"]
    assert tactile["shape"] == [WIDTH, HEIGHT, 3]
    assert (tactile["info"]["video.width"], tactile["info"]["video.height"]) == (HEIGHT, WIDTH)

    original = first_frame(plain, "left_tactile_left")
    transformed = first_frame(turned, "left_tactile_left")
    assert original.shape == (HEIGHT, WIDTH, 3)
    assert transformed.shape == (WIDTH, HEIGHT, 3)
    # A transpose leaves the per-channel means alone, so what is left to compare
    # is the channel order: it is reversed.
    assert np.allclose(
        [transformed[..., channel].mean() for channel in range(3)],
        [original[..., 2 - channel].mean() for channel in range(3)],
        atol=2.0,
    )
    # An untransformed camera in the same run is untouched.
    assert first_frame(turned, "head").shape == (HEIGHT, WIDTH, 3)

    stats = json.loads((turned / "meta" / "stats.json").read_text())
    assert np.array(stats["observation.images.left_tactile_left"]["mean"]).shape == (3, 1, 1)


def test_per_episode_tasks_land_in_the_dataset(recording: Path, tmp_path: Path):
    """A capture schedule gives each session its own task; one string per dataset
    is the exception."""
    import pyarrow.parquet as pq

    config = ConverterConfig(
        fps=FPS,
        gop=4,
        preset="ultrafast",
        task="fallback",
        tasks={"ep_0002": "wipe the plate"},
    )
    assert config.task_for("ep_0002") == "wipe the plate"
    assert config.task_for("ep_0001") == "fallback"

    output = Hdf5ToLeRobotConverter([recording], tmp_path / "tasks", config).convert()

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_tasks"] == 2
    tasks = pq.read_table(output / "meta" / "tasks.parquet").to_pandas()
    assert sorted(tasks.index.tolist()) == ["fallback", "wipe the plate"]

    episodes = pq.read_table(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()
    assert [list(row) for row in episodes["tasks"]] == [["fallback"], ["wipe the plate"]]

    frame = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    index_of = {task: int(idx) for task, idx in tasks["task_index"].items()}
    assert frame["task_index"][:FRAMES].tolist() == [index_of["fallback"]] * FRAMES
    assert frame["task_index"][FRAMES:].tolist() == [index_of["wipe the plate"]] * FRAMES

    sources = json.loads((output / "meta" / "sources.json").read_text())
    assert [entry["task"] for entry in sources["episodes"]] == [
        "fallback",
        "wipe the plate",
    ]


def test_task_map_file_is_read_by_the_cli(tmp_path: Path):
    from hdf52lerobot.cli import load_task_map

    path = tmp_path / "tasks.yaml"
    path.write_text('ep_0001: "pick up the cup"\nep_0002: "fold the towel"\n', encoding="utf-8")
    assert load_task_map(path) == {"ep_0001": "pick up the cup", "ep_0002": "fold the towel"}
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    assert load_task_map(tmp_path / "empty.yaml") == {}
    (tmp_path / "list.yaml").write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_task_map(tmp_path / "list.yaml")


def test_codec_threads_reach_the_encoder(tmp_path: Path):
    """FFmpeg's default sizes every codec instance to the whole machine; this
    setting is what keeps a parallel run from oversubscribing it."""
    from hdf52lerobot.video import VideoAggregator

    config = ConverterConfig(fps=FPS, gop=4, preset="ultrafast", codec_threads=1)
    aggregator = VideoAggregator(tmp_path, "observation.images.cam", WIDTH, HEIGHT, config)
    aggregator._open()
    try:
        assert aggregator._stream.codec_context.thread_count == 1
        assert str(aggregator._stream.codec_context.thread_type) in ("NONE", "ThreadType.NONE", "0")
    finally:
        aggregator.close()

    # 0 keeps FFmpeg's own choice, for a machine running one conversion.
    auto = VideoAggregator(
        tmp_path / "auto", "observation.images.cam", WIDTH, HEIGHT,
        replace(config, codec_threads=0),
    )
    auto._open()
    try:
        assert auto._stream.codec_context.thread_count != 1
    finally:
        auto.close()

    with pytest.raises(ValueError, match="codec_threads"):
        ConverterConfig(codec_threads=-1)


def test_encoder_options_follow_the_encoder():
    """x264 takes CRF and a keyframe floor; NVENC takes a rate-control mode and
    a QP, and rejects the other two."""
    software = ConverterConfig(vcodec="libx264", crf=23, gop=30, preset="veryfast")
    assert software.encoder_options() == {
        "g": "30", "preset": "veryfast", "crf": "23", "keyint_min": "30",
    }
    assert software.is_hardware_encoder is False

    hardware = ConverterConfig(
        vcodec="h264_nvenc", crf=21, gop=30, preset="p4", video_options={"tune": "hq"}
    )
    assert hardware.is_hardware_encoder is True
    assert hardware.encoder_options() == {
        "g": "30", "preset": "p4", "rc": "constqp", "qp": "21", "tune": "hq",
    }
    # video_options has the last word, so a profile can override a derived one.
    assert ConverterConfig(video_options={"crf": "18"}).encoder_options()["crf"] == "18"


def test_info_records_the_format_not_the_encoder():
    """A dataset reads the same whether its video came off the CPU or the GPU."""
    from hdf52lerobot.features import video_info

    for vcodec in ("libx264", "h264_nvenc", "h264_qsv"):
        config = ConverterConfig(vcodec=vcodec)
        assert video_info(640, 480, config)["video.codec"] == "h264"
    assert video_info(640, 480, ConverterConfig(vcodec="hevc_nvenc"))["video.codec"] == "hevc"


def test_gripper_angle_wraps_to_the_calibrated_span():
    """An encoder reads 0..360; a jaw straddling the wrap point is calibrated
    with a negative bound, and its readings must come back to that span."""
    from hdf52lerobot.source import GripperStream

    # Closed at 343.51 degrees (written -16.49), open at 10.91.
    straddling = GripperStream(
        key="left_gripper",
        path=Path("left_umi_087L_openv2_-16.49_10.91.hdf5"),
        timestamps=np.arange(5, dtype=np.float64),
        values=np.array([343.51, 350.0, 0.0, 5.0, 10.91]),
        lo=-16.49,
        hi=10.91,
    )
    assert np.allclose(straddling.unwrapped, [-16.49, -10.0, 0.0, 5.0, 10.91])
    got = straddling.normalized()
    assert got[0] == pytest.approx(0.0)      # 343.51 is closed, not "far past open"
    assert got[-1] == pytest.approx(1.0)
    assert np.all(np.diff(got) > 0)          # and the travel is monotone

    # Without the wrap, 350 would sit 13x past the open end and clip to 1.0.
    naive = np.clip((straddling.values - straddling.lo) / (straddling.hi - straddling.lo), 0, 1)
    assert naive[1] == pytest.approx(1.0)

    # A span that does not straddle the wrap point is untouched.
    plain = GripperStream(
        key="left_gripper",
        path=Path("left_umi_035L_openv2_131.11_158.53.hdf5"),
        timestamps=np.arange(4, dtype=np.float64),
        values=np.array([131.11, 140.0, 158.53, 130.0]),
        lo=131.11,
        hi=158.53,
    )
    assert np.allclose(plain.unwrapped, plain.values)
    assert plain.normalized()[-1] == pytest.approx(0.0)  # just under closed, clipped


def test_gripper_rule_accepts_a_negative_calibration_bound():
    config = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "tacverse_rdt.yaml"
    )
    rule = next(r for r in config.streams if r.kind == "gripper")
    for name, lo, hi in (
        ("left_umi_087L_openv2_-16.49_10.91.hdf5", "-16.49", "10.91"),
        ("right_umi_035R_openv2_247.90_275.38.hdf5", "247.90", "275.38"),
    ):
        resolved = rule.match(name)
        assert resolved is not None, name
        assert (resolved[1]["lo"], resolved[1]["hi"]) == (lo, hi)


def test_channel_bias_and_lookup_match_the_sdk_rule():
    """bias = reference - measured, applied with saturation."""
    from hdf52lerobot.imaging import apply_bias, bias_lookup, channel_bias

    reference = {"red": 142.189, "green": 151.411, "blue": 165.365}
    bias = channel_bias(reference, [120.0, 151.411, 200.0])
    assert bias == pytest.approx([22.189, 0.0, -34.635])

    lookup = bias_lookup(bias)
    assert lookup.shape == (3, 256) and lookup.dtype == np.uint8
    assert lookup[0][100] == 122      # 100 + round(22.189)
    assert lookup[1][100] == 100      # untouched channel
    assert lookup[2][100] == 65       # 100 - round(34.635)
    assert lookup[0][250] == 255      # saturates, never wraps
    assert lookup[2][10] == 0

    frame = np.full((2, 3, 3), 100, dtype=np.uint8)
    out = apply_bias(frame, lookup)
    assert out.dtype == np.uint8
    assert out[0, 0].tolist() == [122, 100, 65]

    # A frame already at the reference is left where it is.
    measured = [142.189, 151.411, 165.365]
    assert channel_bias(reference, measured) == pytest.approx([0, 0, 0])

    with pytest.raises(ValueError, match="missing"):
        channel_bias({"red": 1.0, "green": 2.0}, [1, 2, 3])


def test_rgb_reference_brings_a_camera_to_the_reference_tone(recording: Path, tmp_path: Path):
    """Two cameras with different casts must come out with the same tone."""
    import av

    reference = {"red": 128.0, "green": 128.0, "blue": 128.0}
    base = ConverterConfig(fps=FPS, gop=4, preset="ultrafast", task="t")
    toned = replace(
        base,
        rgb_references={"TEST": reference},
        rgb_bias_frames=10,
        streams=tuple(
            replace(rule, rgb_reference="TEST") if rule.key == "head" else rule
            for rule in base.streams
        ),
    )

    def channel_means(root: Path, key: str, limit: int | None = None) -> np.ndarray:
        path = root / "videos" / f"observation.images.{key}" / "chunk-000" / "file-000.mp4"
        with av.open(str(path)) as container:
            frames = [
                frame.to_ndarray(format="rgb24")
                for position, frame in enumerate(container.decode(video=0))
                if limit is None or position < limit
            ]
        return np.stack(frames).reshape(-1, 3).mean(axis=0)

    plain = Hdf5ToLeRobotConverter([recording], tmp_path / "plain", base).convert()
    fixed = Hdf5ToLeRobotConverter([recording], tmp_path / "fixed", toned).convert()

    # The bias is measured from the opening frames, as the SDK measures at
    # sensor init, so that is where the camera lands on the reference; encoding
    # costs a level or two.
    assert np.abs(channel_means(fixed, "head", 10) - 128.0).max() < 3.0
    assert np.abs(channel_means(plain, "head", 10) - 128.0).max() > 5.0
    # The whole recording moves by that one bias, not per frame.
    shift = channel_means(fixed, "head") - channel_means(plain, "head")
    assert np.allclose(shift, shift[0], atol=1.5)
    # A camera without a reference is untouched.
    assert np.allclose(
        channel_means(plain, "left_wrist"), channel_means(fixed, "left_wrist"), atol=1.0
    )

    # With a channel-swapping transform, the tone must be measured *after* it,
    # or the reference's red would be matched against the frame's blue.
    swapped = replace(
        toned,
        streams=tuple(
            replace(rule, transform=("bgr_to_rgb",), rgb_reference="TEST")
            if rule.key == "head"
            else rule
            for rule in base.streams
        ),
    )
    out = Hdf5ToLeRobotConverter([recording], tmp_path / "swapped", swapped).convert()
    assert np.abs(channel_means(out, "head", 10) - 128.0).max() < 3.0


def test_rgb_reference_must_name_a_known_entry():
    rule = StreamRule(r"^x\.mkv$", "video", "x")
    with pytest.raises(ValueError, match="not in rgb_references"):
        ConverterConfig(streams=(replace(rule, rgb_reference="NOPE"),))
    with pytest.raises(ValueError, match="missing"):
        ConverterConfig(rgb_references={"BAD": {"red": 1.0}})
    with pytest.raises(ValueError, match="only meaningful for 'video'"):
        StreamRule(r"^x\.hdf5$", "pose", "x", rgb_reference="GSPS01")


def test_shipped_config_loads():
    config = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "hdf52lerobot.yaml"
    )
    assert config.fps == 20
    assert config.pair == "index"
    assert config.quat_order == "xyzw"
    assert config.robot_type == "bi_rdt_gripper"
    assert config.master_clock == "left_tcp"
    assert config.state_layout == ("left_tcp", "left_gripper", "right_tcp", "right_gripper")
    assert config.warn_sync_gap_frames < config.max_sync_gap_frames
    kinds = {"left_tcp": "pose", "left_gripper": "gripper", "right_tcp": "pose", "right_gripper": "gripper"}
    assert len(config.state_names(kinds)) == 20


def test_grid_conversion_writes_source_timestamps_and_sources_manifest(recording: Path, tmp_path: Path):
    import pyarrow.parquet as pq

    config = ConverterConfig(
        fps=FPS, pair="grid", keep_source_timestamp=True, task="grid", gop=4, preset="ultrafast"
    )
    output = Hdf5ToLeRobotConverter([recording], tmp_path / "grid", config).convert()

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["observation.source_timestamp"] == {"dtype": "int64", "names": None, "shape": [1]}
    table = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet")
    assert table.column_names[-1] == "observation.source_timestamp"
    stamps = table.column("observation.source_timestamp").to_numpy()
    assert stamps.dtype == np.int64 and stamps.min() > 1_600_000_000 * 10**9
    assert np.all(np.diff(stamps[: info["total_frames"] // 2]) >= 0)
    stats = json.loads((output / "meta" / "stats.json").read_text())
    assert stats["observation.source_timestamp"]["count"] == [info["total_frames"]]
    episodes = pq.read_table(output / "meta" / "episodes" / "chunk-000" / "file-000.parquet").column_names
    assert "stats/observation.source_timestamp/mean" in episodes

    sources = json.loads((output / "meta" / "sources.json").read_text())
    assert sources["format"] == "hdf52lerobot/sources"
    assert sources["config"]["pair"] == "grid" and sources["config"]["fps"] == FPS
    assert [entry["episode_index"] for entry in sources["episodes"]] == [0, 1]
    first = sources["episodes"][0]
    assert first["source_dir"].endswith("ep_0001") and first["source_parent"] == recording.name
    assert first["frames"] == first["source_iterations"] - first["dropped_iterations"] + first["duplicated_frames"]
    assert set(first["tick_offset_ms"]) == {"p50", "p95", "max"}
    assert len(first["files"]) == 11


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


def test_a_stream_can_be_paired_without_deciding_usability():
    """`require_sync: false` keeps the camera and drops only its veto."""
    from hdf52lerobot.source import PoseStream, SourceEpisode, VideoStream

    fps, frames = 20, 40
    clock = np.arange(frames) / fps

    def video(key: str, timestamps: np.ndarray) -> VideoStream:
        return VideoStream(
            key=key,
            path=Path(f"{key}.mkv"),
            timestamps=timestamps,
            width=8,
            height=8,
            frames=len(timestamps),
        )

    # `loose` starts a full second late, far past any sane gap limit.
    episode = SourceEpisode(path=Path("ep"))
    episode.poses["left_tcp"] = PoseStream(
        key="left_tcp",
        path=Path("left_tcp.hdf5"),
        timestamps=clock,
        values=np.tile(np.array([0.0, 0, 0, 1, 0, 0, 0]), (frames, 1)),
    )
    episode.videos["strict"] = video("strict", clock)
    episode.videos["loose"] = video("loose", clock + 1.0)

    base = ConverterConfig(
        fps=fps,
        pair="grid",
        master_clock="left_tcp",
        max_sync_gap_frames=2.0,
        streams=(
            StreamRule(r"^left_tcp$", "pose", "left_tcp"),
            StreamRule(r"^strict$", "video", "strict"),
            StreamRule(r"^loose$", "video", "loose"),
        ),
    )
    with pytest.raises(ValueError, match="loose"):
        build_sync(episode, base)

    relaxed = replace(
        base,
        streams=tuple(
            replace(rule, require_sync=False) if rule.key == "loose" else rule
            for rule in base.streams
        ),
    )
    report = build_sync(episode, relaxed)
    # Still present, still paired, and the drift is reported rather than hidden.
    assert "loose" in report.streams
    assert len(report.streams["loose"].indices) == report.length
    assert any("loose" in message for message in report.unchecked)
    assert not any("strict" in message for message in report.unchecked)

    # A stream nobody wrote a rule for keeps its veto.
    assert relaxed.requires_sync("unknown") is True


def test_require_sync_is_read_from_the_profile():
    config = ConverterConfig.from_yaml(
        Path(__file__).resolve().parent.parent / "configs" / "tacverse_rdt.yaml"
    )
    assert config.requires_sync("side") is False
    assert config.requires_sync("left_wrist") is True
    assert config.requires_sync("left_tcp") is True


def test_sources_records_what_a_normalized_gripper_value_means(dataset: Path):
    """A normalized 1.0 is the calibration window's open end, not the jaw's.

    Without the window in the dataset a consumer can only assume the jaw's
    mechanical range, which draws it wider open than the hardware ever was.
    """
    document = json.loads((dataset / "meta" / "sources.json").read_text())

    summary = document["gripper_normalization"]
    assert "calibration" in summary["definition"]
    assert summary["per_episode_key"] == "episodes[].grippers"

    for record in document["episodes"]:
        for key, calibration in record["grippers"].items():
            assert key.endswith("_gripper")
            assert calibration["travel_deg"] == pytest.approx(
                calibration["open_deg"] - calibration["closed_deg"], abs=1e-3
            )
            assert calibration["travel_rad"] == pytest.approx(
                np.radians(calibration["travel_deg"]), abs=1e-5
            )
            assert calibration["wraps_zero"] is (calibration["closed_deg"] < 0)
            assert key in summary["travel_deg"]
            assert summary["travel_deg"][key]["min"] <= calibration["travel_deg"]
            assert summary["travel_deg"][key]["max"] >= calibration["travel_deg"]


def test_a_wrapping_window_is_flagged_as_such():
    """The five stations whose window straddles zero are worth marking."""
    from hdf52lerobot.source import GripperStream

    wrapping = GripperStream(
        key="left_gripper",
        path=Path("left_umi_032L_openv2_-16.55_10.77.hdf5"),
        timestamps=np.arange(3, dtype=np.float64),
        values=np.array([343.45, 0.0, 10.77]),
        lo=-16.55,
        hi=10.77,
    )
    assert wrapping.lo < 0
    assert wrapping.hi - wrapping.lo == pytest.approx(27.32)
    # And the travel, not the mechanical 40.1 degrees, is what 1.0 maps to.
    assert np.radians(wrapping.hi - wrapping.lo) == pytest.approx(0.4768, abs=1e-4)
