"""Tracker -> EE transform: the frame it produces, and the world step's order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hdf52lerobot.config import ConverterConfig
from hdf52lerobot.geometry import matrices_to_rotation_6d, quaternions_to_matrices
from hdf52lerobot.ee_transform import (
    VIVE_TO_WORLD_R,
    TrackerToEe,
    matrices_to_rotation_6d,
    quaternions_to_matrices,
    transform_pose_block,
)
from hdf52lerobot.geometry import quaternions_to_rotation_6d


def identity_quats(n: int) -> np.ndarray:
    quats = np.zeros((n, 4))
    quats[:, 0] = 1.0
    return quats


def test_default_is_identity_and_leaves_the_pose_alone():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quats = identity_quats(2)
    out_pos, out_rot = transform_pose_block(positions, quats, TrackerToEe(), None)
    assert TrackerToEe().is_identity
    assert np.allclose(out_pos, positions)
    assert np.allclose(out_rot, quaternions_to_rotation_6d(quats))


def test_quaternion_path_matches_the_existing_6d_conversion():
    # A transform-free run must reproduce geometry.py exactly, or every dataset
    # converted before this module would shift.
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(16, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    direct = quaternions_to_rotation_6d(quats)
    through = matrices_to_rotation_6d(quaternions_to_matrices(quats))
    assert np.allclose(direct, through)


def test_approach_axis_becomes_the_eef_z():
    body = TrackerToEe(approach="+z", distance=0.22)
    assert np.allclose(body.rotation()[:, 2], [0, 0, 1])
    assert np.allclose(body.translation(), [0, 0, 0.22])
    # The offset rides with the tracker: rotate the tracker 90 deg about +X and
    # the EE must move along the rotated axis, not the world one.
    rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    quats = _matrix_to_quat(rotation)[None]
    out_pos, _ = transform_pose_block(np.zeros((1, 3)), quats, body, None)
    assert np.allclose(out_pos[0], [0, -0.22, 0], atol=1e-9)


def test_spin_rotates_the_jaw_axis_about_the_approach():
    zero = TrackerToEe(approach="+z", distance=0.2)
    ninety = TrackerToEe(approach="+z", distance=0.2, spin=90.0)
    assert np.allclose(zero.rotation()[:, 2], ninety.rotation()[:, 2])
    assert abs(float(zero.rotation()[:, 0] @ ninety.rotation()[:, 0])) < 1e-9
    # A pure spin cannot move the origin when there is no lateral offset.
    assert np.allclose(zero.translation(), ninety.translation())


def test_world_step_is_a_left_multiply_not_a_conjugation():
    # The trap: `G R G.T` re-labels the body axes too and would rotate the EE
    # frame back out of the convention the body step just put it in.
    body = TrackerToEe(approach="+z", distance=0.3)
    quats = identity_quats(1)
    _, rot6d = transform_pose_block(np.zeros((1, 3)), quats, body, VIVE_TO_WORLD_R)
    approach = rot6d[0, :3]  # first column is X... the third comes from the cross
    matrices = quaternions_to_matrices(quats) @ body.rotation()
    expected = VIVE_TO_WORLD_R @ matrices[0]
    conjugated = VIVE_TO_WORLD_R @ matrices[0] @ VIVE_TO_WORLD_R.T
    assert np.allclose(np.concatenate([expected[:, 0], expected[:, 1]]), rot6d[0])
    assert not np.allclose(np.concatenate([conjugated[:, 0], conjugated[:, 1]]), rot6d[0])
    assert np.allclose(approach, expected[:, 0])


def test_world_step_sends_steamvr_up_to_z_up():
    # SteamVR +Y is up; after the rotation the up axis must be +Z.
    assert np.allclose(VIVE_TO_WORLD_R @ np.array([0.0, 1.0, 0.0]), [0, 0, 1])
    assert abs(np.linalg.det(VIVE_TO_WORLD_R) - 1.0) < 1e-12


def test_config_rejects_an_unknown_world_frame_or_stream():
    with pytest.raises(ValueError, match="world_frame"):
        ConverterConfig(world_frame="y_up")
    with pytest.raises(ValueError, match="tracker_to_ee"):
        ConverterConfig(tracker_to_ee={"nope_tcp": TrackerToEe(distance=0.1)})


def test_config_parses_the_profile_mapping():
    config = ConverterConfig.from_dict(
        {
            "state_layout": ["left_tcp", "left_gripper"],
            "world_frame": "vive_to_xense",
            "pose_transforms": {"left_tcp": {"approach": "+z", "distance": 0.22, "spin": 90}},
        }
    )
    body = config.tracker_to_ee_for("left_tcp")
    assert body.distance == 0.22 and body.spin == 90 and body.measured is False
    assert not body.is_identity
    assert config.tracker_to_ee_for("left_gripper").is_identity
    assert config.world_matrix() is not None


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    scale = np.sqrt(trace + 1.0) * 2.0
    return np.array(
        [
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    )


def test_the_old_key_name_still_loads(tmp_path: Path):
    """`pose_transforms` was this key's first name; profiles in the wild use it."""
    profile = tmp_path / "old.yaml"
    profile.write_text(
        "robot_type: bi_rdt_gripper\n"
        "pose_transforms:\n"
        "  left_tcp: {approach: '+z', distance: 0.22, measured: false}\n"
    )
    config = ConverterConfig.from_yaml(profile)
    assert config.tracker_to_ee_for("left_tcp").distance == 0.22

    both = tmp_path / "both.yaml"
    both.write_text(
        "robot_type: bi_rdt_gripper\n"
        "tracker_to_ee: {left_tcp: {distance: 0.1}}\n"
        "pose_transforms: {left_tcp: {distance: 0.2}}\n"
    )
    with pytest.raises(ValueError, match="not both"):
        ConverterConfig.from_yaml(both)
