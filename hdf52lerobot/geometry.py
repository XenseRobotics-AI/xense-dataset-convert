"""Rotation conversions matching the LeRobot ``bi_taccap_gripper`` convention.

The 6D representation is the first two columns of the rotation matrix, the same
as ``lerobot.utils.robot_utils.quaternion_to_rotation_6d``. Keeping a local copy
avoids a runtime dependency on lerobot for a dozen lines of algebra.
"""

from __future__ import annotations

import numpy as np

from .config import QUAT_ORDERS


def to_wxyz(quaternions: np.ndarray, order: str) -> np.ndarray:
    """Reorder quaternion components into ``[w, x, y, z]``.

    Args:
        quaternions: Array of shape ``(..., 4)``.
        order: Component order of the input, ``"wxyz"`` or ``"xyzw"``.
    """
    if order not in QUAT_ORDERS:
        raise ValueError(f"quaternion order must be one of {QUAT_ORDERS}, got {order!r}")
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError(f"expected trailing dimension 4, got {quaternions.shape}")
    if order == "wxyz":
        return quaternions
    return quaternions[..., [3, 0, 1, 2]]


def quaternions_to_rotation_6d(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert ``[w, x, y, z]`` quaternions to the 6D rotation representation.

    Args:
        quaternions_wxyz: Array of shape ``(N, 4)``. Need not be normalized.

    Returns:
        Array of shape ``(N, 6)``: ``[r1, r2, r3]`` is the first column of the
        rotation matrix, ``[r4, r5, r6]`` the second.
    """
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"expected shape (N, 4), got {quaternions.shape}")

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encountered a zero-norm quaternion")
    w, x, y, z = (quaternions / norms).T

    first_column = np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)], axis=1
    )
    second_column = np.stack(
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)], axis=1
    )
    return np.concatenate([first_column, second_column], axis=1)


def rotation_6d_to_quaternions(rotations_6d: np.ndarray) -> np.ndarray:
    """Invert :func:`quaternions_to_rotation_6d` (Gram-Schmidt, then to wxyz).

    Only used for verification; the converter itself never needs it.
    """
    rotations = np.asarray(rotations_6d, dtype=np.float64)
    if rotations.ndim != 2 or rotations.shape[1] != 6:
        raise ValueError(f"expected shape (N, 6), got {rotations.shape}")

    a1, a2 = rotations[:, :3], rotations[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    matrices = np.stack([b1, b2, b3], axis=2)  # columns

    trace = np.trace(matrices, axis1=1, axis2=2)
    quaternions = np.empty((len(matrices), 4), dtype=np.float64)
    for index, (matrix, tr) in enumerate(zip(matrices, trace)):
        if tr > 0:
            scale = np.sqrt(tr + 1.0) * 2.0
            quaternions[index] = [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        else:
            diagonal = int(np.argmax(np.diagonal(matrix)))
            i, j, k = diagonal, (diagonal + 1) % 3, (diagonal + 2) % 3
            scale = np.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
            components = np.zeros(3)
            components[i] = 0.25 * scale
            components[j] = (matrix[j, i] + matrix[i, j]) / scale
            components[k] = (matrix[k, i] + matrix[i, k]) / scale
            quaternions[index] = [
                (matrix[k, j] - matrix[j, k]) / scale,
                components[0],
                components[1],
                components[2],
            ]

    negative = quaternions[:, 0] < 0
    quaternions[negative] *= -1.0
    return quaternions
