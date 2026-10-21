"""Sidecar v2 / v4 compaction: ported from sidecar_v2.rs's own tests, plus the golden round trip."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mcap2lerobot import xtac_pack as X
from mcap2lerobot.xtac_sidecar import (
    compact_collection_devices,
    compact_taccap_extrinsics,
    expand_collection_devices,
    expand_taccap_extrinsics,
)

GOLDEN = Path(os.environ.get("XTAC_PARITY_GOLDEN", Path.home() / "taccap-cross" / "parity-golden"))


def _device_entry(index: int, anchor_x: float, sn: str) -> dict:
    return {
        "episode_index": index, "source_episode_id": f"EP-{index}", "legacy": False,
        "devices": {"collector": {"software_version": "0.4.6"},
                    "pico": {"serial_number": sn, "camera_parameters": {"raw_parameters": {"big": [1, 2, 3]}}}},
        "export": {
            "exporter": "taccap-collector", "exporter_version": "0.4.6", "world_frame": "head_yaw_at_episode_start",
            "pose_interp_fallback_rows": {},
            "world_anchor": {"position": [anchor_x, 0.0, 0.0], "yaw_deg": 0.1, "time_ns": index},
            "initial_pose_check": {"status": "unconfigured", "bounds": {"frames": 1},
                                   "left": {"position": [0.3, 0.2, 0.0]}},
        },
    }


def test_identical_blocks_are_stored_once_and_expand_back_exactly():
    entries = [_device_entry(0, 0.5, "A"), _device_entry(1, -0.3, "A"), _device_entry(2, 0.1, "B")]
    file = compact_collection_devices(entries)
    assert file["version"] == 2
    assert (len(file["devices"]), len(file["exports"]), len(file["initial_pose_bounds"])) == (2, 1, 1)
    assert [e["devices"] for e in file["episodes"]] == ["devices_0", "devices_0", "devices_1"]
    assert file["episodes"][1]["export"] == "export_0"
    assert file["episodes"][1]["world_anchor"]["position"][0] == -0.3
    assert file["episodes"][1]["initial_pose_check"]["bounds"] == "initial_pose_bounds_0"
    assert (file["total_episodes"], file["legacy_episodes"]) == (3, 0)
    assert expand_collection_devices(file) == entries
    assert expand_collection_devices({"version": 1, "episodes": entries}) == entries     # v1 passes through


def test_legacy_entries_without_export_round_trip():
    legacy = {"episode_index": 0, "source_episode_id": None, "legacy": True, "devices": {"collector": None}}
    file = compact_collection_devices([legacy])
    assert file["legacy_episodes"] == 1 and file["exports"] == {}
    assert expand_collection_devices(file) == [legacy]


def test_extrinsics_profiles_dedupe_and_keep_warnings_inline():
    def entry(index: int, warning: str) -> dict:
        return {"episode_index": index, "source_episode_id": f"EP-{index}", "tracker_bindings": {"left": "L"},
                "tracker_to_ee": {"version": 1}, "gripper_open_rad": {"left": 1.2, "right": 1.1},
                "legacy_extrinsics": False, "legacy_world_pose": False, "poses_are_ee": True,
                "warnings": [warning] if warning else []}

    entries = [entry(0, ""), entry(1, "x")]
    file = compact_taccap_extrinsics(entries)
    assert file["version"] == 4 and len(file["profiles"]) == 1
    assert file["episodes"][1] == {"episode_index": 1, "source_episode_id": "EP-1", "warnings": ["x"],
                                   "profile": "extrinsics_0"}
    assert expand_taccap_extrinsics(file) == entries
    assert expand_taccap_extrinsics({"version": 3, "episodes": entries}) == entries


def test_ids_count_past_ten_in_first_appearance_order():
    entries = [_device_entry(i, 0.0, f"S{i}") for i in range(12)]
    file = compact_collection_devices(entries)
    assert file["episodes"][11]["devices"] == "devices_11"
    assert expand_collection_devices(file) == entries


def test_compaction_does_not_mutate_its_input():
    entries = [_device_entry(0, 0.5, "A")]
    before = json.dumps(entries)
    compact_collection_devices(entries)
    assert json.dumps(entries) == before


def test_unknown_reference_is_an_error():
    file = compact_collection_devices([_device_entry(0, 0.5, "A")])
    file["episodes"][0]["devices"] = "devices_9"
    with pytest.raises(ValueError, match="unknown devices id devices_9"):
        expand_collection_devices(file)


@pytest.mark.parametrize("scenario", ["dual_claw_head-lerobot", "legacy_no_sidecar-lerobot"])
def test_golden_files_expand_to_the_training_mcap_entries(scenario):
    root = GOLDEN / scenario
    if not (root / "train_mcap").is_dir():
        pytest.skip(f"parity golden {root} not present")
    episodes = X.order_episodes([X.read_train_mcap(p) for p in sorted((root / "train_mcap").glob("*.train.mcap"))])
    meta = root / "lerobot" / "meta"
    devices = expand_collection_devices(json.loads((meta / "xumi_collection_devices.json").read_text()))
    extrinsics = expand_taccap_extrinsics(json.loads((meta / "taccap_extrinsics.json").read_text()))
    assert devices == [{**e.devices_entry, "episode_index": i} for i, e in enumerate(episodes)]
    assert extrinsics == [{**e.extrinsics_entry, "episode_index": i} for i, e in enumerate(episodes)]
