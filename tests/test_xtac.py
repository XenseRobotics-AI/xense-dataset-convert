"""Unit tests for the XTacData dialect (mcap2lerobot/xtac.py).

These cover the pure-Python pieces — topic mapping, tracker-confidence
extraction, metadata synthesis and the extra ``meta/*.json`` writer — that need
neither lerobot, PyAV nor a real MCAP. The two-pass decode over a real recording
is exercised out of tree against the XTacData fixtures.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from mcap2lerobot.config import DIALECT_XTAC, ConverterConfig
from mcap2lerobot.converter import McapToLeRobotConverter, build_features, plan_pairing
from mcap2lerobot.reader import EpisodeData, TimedArray, VideoStreamIndex
from mcap2lerobot.xtac import (
    TRACKER_CONFIDENCE_FEATURE,
    XTacMcapReader,
    _tracker_confidence,
    write_xtac_metadata,
)
from mcap2lerobot.xtac_sidecar import (
    expand_collection_devices,
    expand_taccap_extrinsics,
)

HEAD = "observation.images.head_left"
TACTILE = "observation.images.left_tactile_left"


def test_default_dialect_is_xtac():
    assert ConverterConfig().dialect == DIALECT_XTAC


def test_unknown_dialect_is_rejected():
    with pytest.raises(ValueError, match="Unknown dialect"):
        ConverterConfig(dialect="nonsense")


def test_video_topic_maps_the_eight_cameras():
    reader = XTacMcapReader(ConverterConfig())
    assert reader.video_feature("/camera/head/left/h264") == HEAD
    assert reader.video_feature("camera/left/fisheye/h264") == "observation.images.left_wrist"
    assert reader.video_feature("/camera/left/tactile_2/h264") == "observation.images.left_tactile_right"
    assert reader.video_feature("/camera/unknown/h264") is None


def test_tracker_confidence_orders_left_then_right_by_serial():
    pose = {
        "joints": [
            {"sn": "R-SERIAL", "side": "right"},
            {"sn": "L-SERIAL", "side": "left"},
        ],
        "state": {
            "Motion": {
                "joints": [
                    {"sn": "R-SERIAL", "confidence": 1},
                    {"sn": "L-SERIAL", "confidence": 3},
                ]
            }
        },
    }
    value = _tracker_confidence(pose)
    assert value.tolist() == [3.0, 1.0]
    assert value.dtype == np.float32


def test_tracker_confidence_missing_a_side_returns_none():
    pose = {
        "joints": [{"sn": "L-SERIAL", "side": "left"}],
        "state": {"Motion": {"joints": [{"sn": "L-SERIAL", "confidence": 1}]}},
    }
    assert _tracker_confidence(pose) is None


def _episode_with_metadata(task: str | None = None) -> tuple[XTacMcapReader, EpisodeData]:
    names = ",".join(f"d{i}" for i in range(29))
    episode = EpisodeData(source=Path("EP.mcap"))
    episode.states = [TimedArray(0, np.zeros(29, dtype=np.float64), task_index=0)]
    episode.actions = [TimedArray(0, np.zeros(29, dtype=np.float64))]
    episode.videos = {
        HEAD: VideoStreamIndex(HEAD, "/camera/head/left/h264", np.array([0]), (480, 640, 3), "h264")
    }
    episode.extra_vectors = {TRACKER_CONFIDENCE_FEATURE: [TimedArray(0, np.ones(2, dtype=np.float32))]}
    records = {
        "xense_train_mcap": {"dimension_names": names, "fps": "30", "state_dim": "29"},
        "episode": {"task_id": "TASK-1", "episode_id": "EP-1"},
    }
    reader = XTacMcapReader(ConverterConfig(task=task))
    reader._synthesize_metadata(episode, records)
    return reader, episode


def test_synthesize_metadata_builds_info_and_episode_meta():
    _, episode = _episode_with_metadata()
    assert episode.info["fps"] == 30
    assert episode.info["robot_type"] == "xtac_umi_g1"
    state = episode.info["features"]["observation.state"]
    assert state["dtype"] == "float32" and state["shape"] == [29]
    assert state["names"][0] == "d0" and state["names"][-1] == "d28"
    assert episode.info["features"][HEAD]["info"]["video.codec"] == "h264"
    assert TRACKER_CONFIDENCE_FEATURE in episode.info["features"]
    # task falls back to the source task_id when not overridden
    assert episode.episode_meta["tasks"] == ["TASK-1"]


def test_synthesize_metadata_warns_when_dimension_names_do_not_fit(caplog):
    """A name list that does not match the state size is dropped — loudly.

    Everything downstream addresses these columns by name, so numbering them
    instead is the difference between a readable dataset and an opaque one. It
    is allowed to happen; it is not allowed to happen quietly.
    """
    episode = EpisodeData(source=Path("EP.mcap"))
    episode.states = [TimedArray(0, np.zeros(29, dtype=np.float64), task_index=0)]
    episode.actions = [TimedArray(0, np.zeros(29, dtype=np.float64))]
    records = {"xense_train_mcap": {"dimension_names": "a,b,c", "fps": "30"}, "episode": {}}

    with caplog.at_level(logging.WARNING, logger="mcap2lerobot.xtac"):
        XTacMcapReader(ConverterConfig())._synthesize_metadata(episode, records)

    names = episode.info["features"]["observation.state"]["names"]
    assert len(names) == 29 and names[0] == "state_0"
    assert "3 dimension_names" in caplog.text


def test_synthesize_metadata_honours_task_override():
    _, episode = _episode_with_metadata(task="do the thing")
    assert episode.episode_meta["tasks"] == ["do the thing"]


def test_build_features_and_plan_include_extra_vectors():
    _, episode = _episode_with_metadata()
    features = build_features(episode)
    assert features[TRACKER_CONFIDENCE_FEATURE] == {
        "dtype": "float32",
        "shape": (2,),
        "names": ["left_tracker.confidence", "right_tracker.confidence"],
    }
    plan = plan_pairing(episode, fps=30, max_sync_gap_frames=1.5)
    assert plan.extra_index[TRACKER_CONFIDENCE_FEATURE].tolist() == [0]


class _FakeReader:
    def __init__(self, episode):
        self.episode = episode

    def read(self, path):
        return self.episode

    def iter_frames(self, episode, video_index):
        for position in range(len(episode.states)):
            yield {name: np.zeros((2, 2, 3), np.uint8) for name in video_index}


class _FakeDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.episodes = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []

    def finalize(self):
        pass


def test_extra_vector_column_is_written_to_every_frame(tmp_path):
    _, episode = _episode_with_metadata()
    episode.source = tmp_path / "input" / "EP.mcap"
    episode.source.parent.mkdir()
    episode.source.touch()
    episode.source_metadata = None  # skip the dialect metadata writer for this unit test

    dataset = _FakeDataset()
    McapToLeRobotConverter(
        inputs=[episode.source.parent],
        output_root=tmp_path / "out",
        repo_id="local/test",
        config=ConverterConfig(task="t"),
        reader=_FakeReader(episode),
        dataset_factory=lambda **kwargs: dataset,
    ).convert()

    frame = dataset.episodes[0][0]
    assert frame[TRACKER_CONFIDENCE_FEATURE].tolist() == [1.0, 1.0]
    assert frame[TRACKER_CONFIDENCE_FEATURE].dtype == np.float32


# -- the extra meta writer ----------------------------------------------------


TRACKER_TO_EE = {
    "version": 1,
    "left": {"tracker_sn": None, "position_m": [-0.16, -0.10, 0.02], "quaternion_wxyz": [0.1, -0.3, 0.9, -0.05]},
    "right": {"tracker_sn": None, "position_m": [-0.16, 0.10, 0.02], "quaternion_wxyz": [0.1, 0.3, 0.9, 0.05]},
}


def _episode_record(episode_id: str, missing: list[str] | None = None, extrinsic: bool = True) -> dict:
    devices = {
        "cameras": [{"role": "fisheye", "serial_number": "CAM", "side": "left"}],
        "trackers": {"left": {"serial_number": "L-SN"}, "right": {"serial_number": "R-SN"}},
        "missing_fields": missing or [],
    }
    calibration = {
        "cameras": {
            "/camera/left/tactile_1/h264": {
                "rectify_kind": "tactile",
                "rectified": True,
                "rectify_config": {"base_grid": [1, 2, 3, 4], "rows": 4, "cols": 2},
            }
        }
    }
    record: dict = {
        "episode": {
            "episode_id": episode_id,
            "left_gripper_open_rad": "1.10",
            "right_gripper_open_rad": "1.20",
            "xumi.collection_devices.v1": json.dumps(devices),
            "xumi.camera_calibration.v1": json.dumps(calibration),
        }
    }
    if extrinsic:
        record["pico_channel"] = {"pose_stage": "ee", "tracker_to_ee": json.dumps(TRACKER_TO_EE)}
    return record


def test_write_xtac_metadata_produces_the_provenance_files(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {"features": {TACTILE: {"dtype": "video", "shape": [400, 700, 3], "info": {"video.codec": "h264"}}}}
        )
    )

    records = [_episode_record("EP-A"), _episode_record("EP-B", missing=["cameras"])]
    write_xtac_metadata(tmp_path, records, robot_type="xtac_umi_g1")

    hardware = json.loads((meta / "hardware.json").read_text())
    assert hardware == {"robot_type": "xtac_umi_g1", "epochs": []}

    devices = json.loads((meta / "xumi_collection_devices.json").read_text())
    assert devices["version"] == 2
    assert devices["total_episodes"] == 2
    assert devices["snapshot_episodes"] == 1 and devices["legacy_episodes"] == 1
    assert devices["episodes"][0]["legacy"] is False
    assert devices["episodes"][1]["legacy"] is True
    assert devices["episodes"][0]["source_episode_id"] == "EP-A"
    assert expand_collection_devices(devices)[0]["devices"] == devices["devices"]["devices_0"]

    extrinsics = json.loads((meta / "taccap_extrinsics.json").read_text())
    assert extrinsics["version"] == 4
    first = expand_taccap_extrinsics(extrinsics)[0]
    assert first["poses_are_ee"] is True
    assert first["legacy_extrinsics"] is False
    assert first["tracker_to_ee"] == TRACKER_TO_EE
    assert first["tracker_bindings"] == {"left": "L-SN", "right": "R-SN"}
    assert first["gripper_open_rad"] == {"left": 1.10, "right": 1.20}

    info = json.loads((meta / "info.json").read_text())
    tactile_info = info["features"][TACTILE]["info"]
    assert not [k for k in tactile_info if k.startswith("taccap.")]   # tactile recipe no longer written
    assert tactile_info["video.codec"] == "h264"  # left intact


def test_taccap_extrinsics_omits_transform_when_channel_metadata_absent(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"features": {}}))
    write_xtac_metadata(tmp_path, [_episode_record("EP-A", extrinsic=False)], robot_type="xtac_umi_g1")
    first = expand_taccap_extrinsics(json.loads((meta / "taccap_extrinsics.json").read_text()))[0]
    assert first["tracker_to_ee"] is None
    assert first["legacy_extrinsics"] is True
    assert first["poses_are_ee"] is False
