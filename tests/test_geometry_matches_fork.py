"""The ported conversions must agree with the fork they were ported from.

:mod:`hdf52lerobot.geometry` carries the conventions of
``lerobot.utils.robot_utils`` in xense-taccap-lerobot, ported rather than
imported so these converters run without lerobot. A convention that drifts is
silent -- a 6D rotation laid out by rows instead of columns still has six
plausible numbers in it -- so it is pinned here against the real thing.

The fork is not a test dependency: every test skips when it cannot be imported,
which is the case in CI. Run them locally, where the checkout exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

FORK_SOURCE = Path.home() / "lerobot-xense" / "src"


def fork_utils():
    """The fork's ``robot_utils``, or skip.

    Imported off its source tree rather than an installed package: the point is
    to check against the checkout being developed alongside this repo.
    """
    if not (FORK_SOURCE / "lerobot" / "utils" / "robot_utils.py").exists():
        pytest.skip(f"xense-taccap-lerobot not checked out at {FORK_SOURCE}")
    if str(FORK_SOURCE) not in sys.path:
        sys.path.insert(0, str(FORK_SOURCE))
    try:
        from lerobot.utils import robot_utils
    except Exception as error:  # torch missing, version skew, anything
        pytest.skip(f"cannot import the fork's robot_utils: {error}")
    return robot_utils


@pytest.fixture(scope="module")
def poses() -> tuple[np.ndarray, np.ndarray]:
    """Random unit quaternions and positions, plus the awkward special cases."""
    rng = np.random.default_rng(20260916)
    quaternions = rng.normal(size=(64, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    special = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # identity
            [0.0, 1.0, 0.0, 0.0],  # 180 deg about x, where trace is smallest
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],  # the other half of the double cover
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
    quaternions = np.concatenate([special, quaternions])
    positions = rng.normal(size=(len(quaternions), 3))
    return positions, quaternions


def test_quaternion_to_matrix_agrees(poses):
    fork = fork_utils()
    from hdf52lerobot.geometry import quaternions_to_matrices

    positions, quaternions = poses
    ours = quaternions_to_matrices(quaternions)
    theirs = np.stack(
        [
            fork.quaternion_to_matrix(np.concatenate([p, q]), "wxyz")[:3, :3]
            for p, q in zip(positions, quaternions)
        ]
    )
    assert np.abs(ours - theirs).max() < 1e-6


def test_rotation_6d_agrees_and_is_column_major(poses):
    """The 6D layout is the first two *columns*. Rows would be a silent corruption."""
    fork = fork_utils()
    from hdf52lerobot.geometry import matrices_to_rotation_6d, quaternions_to_matrices

    _, quaternions = poses
    matrices = quaternions_to_matrices(quaternions)
    ours = matrices_to_rotation_6d(matrices)
    theirs = np.stack([fork.quaternion_to_rotation_6d(*q) for q in quaternions])
    assert np.abs(ours - theirs).max() < 1e-6

    # Columns, explicitly -- and rows would not have passed above either.
    assert np.allclose(ours[:, :3], matrices[:, :, 0])
    assert np.allclose(ours[:, 3:], matrices[:, :, 1])


def test_pose_7d_to_9d_agrees(poses):
    fork = fork_utils()
    from hdf52lerobot.geometry import poses_7d_to_9d

    positions, quaternions = poses
    block = np.concatenate([positions, quaternions], axis=1)
    ours = poses_7d_to_9d(block)
    theirs = np.stack([fork.pose7d_to_pose9d(row, "wxyz") for row in block])
    assert np.abs(ours - theirs).max() < 1e-5


def test_pose_9d_to_7d_agrees(poses):
    """Compared as rotations: the fork and we may pick opposite signs for q."""
    fork = fork_utils()
    from hdf52lerobot.geometry import poses_7d_to_9d, poses_9d_to_7d, quaternions_to_matrices

    positions, quaternions = poses
    nine = poses_7d_to_9d(np.concatenate([positions, quaternions], axis=1))
    ours = poses_9d_to_7d(nine)
    theirs = np.stack([fork.pose9d_to_pose7d(row, "wxyz") for row in nine])

    assert np.abs(ours[:, :3] - theirs[:, :3]).max() < 1e-5
    assert np.abs(
        quaternions_to_matrices(ours[:, 3:]) - quaternions_to_matrices(theirs[:, 3:])
    ).max() < 1e-5


def test_quaternion_multiply_agrees(poses):
    fork = fork_utils()
    from hdf52lerobot.geometry import multiply_quaternions

    _, quaternions = poses
    left, right = quaternions[:-1], quaternions[1:]
    ours = multiply_quaternions(left, right)
    theirs = np.stack([fork.quaternion_multiply(a, b) for a, b in zip(left, right)])
    assert np.abs(ours - theirs).max() < 1e-6


def test_slerp_agrees_including_the_near_parallel_fallback(poses):
    fork = fork_utils()
    from hdf52lerobot.geometry import slerp_quaternions

    _, quaternions = poses
    start, end = quaternions[:-1], quaternions[1:]
    for t in (0.0, 0.25, 0.5, 1.0):
        fractions = np.full(len(start), t)
        ours = slerp_quaternions(start, end, fractions)
        theirs = np.stack([fork.slerp_quaternion(a, b, t, "wxyz") for a, b in zip(start, end)])
        # Both take the shorter arc, so any sign difference is the double cover.
        aligned = np.where(np.sum(ours * theirs, axis=1, keepdims=True) < 0, -theirs, theirs)
        assert np.abs(ours - aligned).max() < 1e-5, f"at t={t}"

    # A pair a hair apart takes the linear fallback in both.
    nudged = start[:4] + 1e-5
    ours = slerp_quaternions(start[:4], nudged, np.full(4, 0.5))
    theirs = np.stack([fork.slerp_quaternion(a, b, 0.5, "wxyz") for a, b in zip(start[:4], nudged)])
    assert np.abs(ours - theirs).max() < 1e-5


def test_normalize_agrees_on_both_orders_and_on_a_dead_quaternion(poses):
    fork = fork_utils()
    from hdf52lerobot.geometry import normalize_quaternions

    _, quaternions = poses
    unnormalised = quaternions[:8] * np.array([[0.3], [7.0], [1.0], [0.01], [2.0], [1.0], [5.0], [0.5]])
    for order in ("wxyz", "xyzw"):
        ours = normalize_quaternions(unnormalised, order)
        theirs = np.stack([fork.normalize_quaternion(q, order) for q in unnormalised])
        assert np.abs(ours - theirs).max() < 1e-6, order

    # Zero norm becomes the identity in both, rather than dividing by zero.
    assert np.allclose(normalize_quaternions(np.zeros((2, 4)))[:, 0], 1.0)
    assert np.allclose(normalize_quaternions(np.zeros((2, 4)))[:, 1:], 0.0)
