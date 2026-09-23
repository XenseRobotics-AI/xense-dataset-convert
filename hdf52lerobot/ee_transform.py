"""Tracker → EE rigid transform and world frame for the UMI/Vive rig.

Why this exists
---------------
What the recorder writes for ``{side}_tcp`` is the **HTC Vive tracker's** pose,
not a TCP: ``*_tracker_LHR-*.hdf5`` reproduces in the dataset bit-for-bit
(position identical, rotation to 2e-8) when no transform is configured here.
Two things are therefore missing from a raw conversion, and both are properties
of the capture rig rather than of any consumer:

* the body-fixed **tracker → two-finger midpoint** offset, and
* the **world frame**: SteamVR is Y-up, the Xense convention is X forward,
  Y left, Z up.

The TacCap rig applies both upstream — ``Pico4TrackerReader`` conjugates the
world frame and ``robots/taccap_gripper/ee_transform.py`` right-multiplies a
CAD-measured ``^Tracker T_EE`` before its HDF5 is written — which is why that
path needs nothing here. The RDT capture stack, inherited from RDT2/UMI, does
neither: upstream never needed a world frame because their model only ever sees
poses re-expressed relative to the current frame, so nothing there notices that
the world is Y-up.

Order, and the trap in it
-------------------------
::

    R_eef, t_eef   = R_tracker @ R_body,  p_tracker + R_tracker @ t_body
    R_out, t_out   = G @ R_eef,           G @ t_eef

The body transform is applied in **raw tracker axes**, so its numbers mean what
a tape measure on the rig means. The world step is then a **left-multiply, not
a conjugation**: ``G · R · Gᵀ`` re-labels the body axes as well as the reference
frame, which would rotate the EE frame straight back out of the convention the
first step put it in. (The TacCap code conjugates because there the body frame
is the device's own and is meant to be re-labelled; its ``APPLY_G_REBASE``
section is the same trap seen from the other side.)

``G`` is exact — a convention, not a calibration. It fixes gravity only:
SteamVR's yaw and origin are set at room-setup time, so the workspace still
faces an arbitrary direction, sits at an arbitrary height, and differs between
capture sessions.

The frame that comes out
------------------------
``+X`` forward along the approach, ``+Y`` left, ``+Z`` up, origin at the
two-finger midpoint — the same convention as the world frame above, and the one
every UMI dataset in this corpus is defined in. TacCap already records it, so
one reading serves both rigs and anything keyed off these columns can stay
rig-agnostic.

It is **not** the vendor URDF's own frame, which approaches along its ``+Z``
(jaws at ``z = +0.2572`` from ``base_link``), nor RDT2's deployment frame, which
puts the fingertips at ``±width/2`` in X and ``±thickness/2`` in Y at ``z = 0``.
Both of those are model-side conventions; a consumer that draws the vendor model
reconciles them with a fixed axis cycle it has to carry anyway, and the viewer
already does. Choosing the model's frame here would spare that one consumer a
constant and cost every other consumer the corpus-wide convention.

Provisional values
------------------
``distance`` and ``spin`` are **not measured**. No CAD of the handle and no
hardware were available, so what the pose data itself supports is:

* the approach axis is the tracker's ``+Z`` — across two datasets, both hands,
  four episodes each, it lies within 45° of vertical-down in 58-81% of frames
  while ±X and ±Y never do once, and a full 3-parameter pivot calibration puts
  the offset's z component on top in every fit;
* ``|offset|`` is **0.18-0.24 m** by those pivot fits — but they disagree with
  each other by 5-15 cm while each is internally precise to 3-9 mm, so each is
  locking onto a real pivot that is not the fingertip (the held object, the
  wrist). Precision there is not accuracy;
* the **spin is not observable at all** from pose data: which way the jaws open
  leaves no trace in a trajectory. A parallel jaw is symmetric under 180°, so it
  is a choice between two values, not four.

So the shipped numbers are a placeholder that reads correctly at the scale of the
scene and will be wrong in detail. ``measured`` stays ``False`` until they come
from the rig, and it is recorded in ``meta/sources.json`` with everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import matrices_to_rotation_6d, quaternions_to_matrices

# SteamVR/OpenVR (X right, Y up, -Z forward) -> Xense world (X forward, Y left,
# Z up); position (x, y, z) -> (-z, -x, y). Numerically identical to
# `PICO_TO_WORLD_R` in xense-taccap-lerobot: the two devices state the same
# convention, so the same matrix serves both. Keep one definition per repo -- a
# second, drifting copy of G is the failure mode that file also warns about.
VIVE_TO_WORLD_R = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

WORLD_FRAMES: dict[str, np.ndarray | None] = {
    "none": None,
    "vive_to_xense": VIVE_TO_WORLD_R,
}

AXES: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


@dataclass(frozen=True)
class TrackerToEe:
    """``^Tracker T_EE`` for one pose stream.

    Args:
        approach: Tracker axis pointing from the tracker at the jaws. Becomes
            the EE frame's ``+X``, the corpus-wide forward. ``None`` means no
            transform at all — see :meth:`rotation`.
        distance: Tracker origin -> two-finger midpoint along that axis, metres.
        lateral: Offsets perpendicular to it, along the EE's Y and Z, metres.
        spin: Rotation about the approach axis in degrees; sets which way the
            jaws open. Only two values are distinct for a parallel jaw.
        measured: True once these came from the rig rather than from the
            placeholder in this module's docstring.
    """

    approach: str | None = None
    distance: float = 0.0
    lateral: tuple[float, float] = (0.0, 0.0)
    spin: float = 0.0
    measured: bool = False

    def __post_init__(self) -> None:
        if self.approach is not None and self.approach not in AXES:
            raise ValueError(f"approach must be one of {sorted(AXES)}, got {self.approach!r}")
        if self.approach is None and (self.distance or any(self.lateral) or self.spin):
            raise ValueError(
                "approach is required to place an offset: without it there is no axis "
                "for `distance` to run along and no axis for `spin` to turn about"
            )
        if len(self.lateral) != 2:
            raise ValueError(f"lateral must be two numbers, got {self.lateral!r}")

    @property
    def is_identity(self) -> bool:
        """True when this genuinely leaves the recorded pose alone.

        Read off the matrices rather than the fields: a rotation that is not
        the identity is exactly the silent failure this class exists to avoid,
        and a field-by-field check drifts from the maths the moment the seed
        convention in :meth:`rotation` is touched.
        """
        return bool(
            np.allclose(self.rotation(), np.eye(3), atol=1e-12)
            and np.allclose(self.translation(), 0.0, atol=1e-12)
        )

    def rotation(self) -> np.ndarray:
        """EE axes expressed in tracker axes, as columns.

        ``+X`` is the approach; Y and Z are free up to a rotation about it,
        which is what ``spin`` sets. The seed is the tracker's own Y projected
        perpendicular to the approach, chosen for stability rather than for
        meaning — with the device free in the operator's hand there is no
        absolute "left" to align to, so the jaw plane comes from ``spin``.

        **No approach declared means the identity.** The default instance has
        to be a true no-op: it is what every stream without a `tracker_to_ee`
        entry gets, and a frame quietly rotated by whichever seed was convenient
        is exactly the silent failure this module exists to prevent.
        """
        if self.approach is None:
            return np.eye(3)
        x_axis = AXES[self.approach]
        seed = AXES["+y"] if abs(float(x_axis[1])) < 0.9 else AXES["+z"]
        y_seed = seed - float(seed @ x_axis) * x_axis
        y_seed /= np.linalg.norm(y_seed)
        z_seed = np.cross(x_axis, y_seed)
        angle = np.radians(self.spin)
        y_axis = np.cos(angle) * y_seed + np.sin(angle) * z_seed
        z_axis = np.cross(x_axis, y_axis)
        return np.stack([x_axis, y_axis, z_axis], axis=1)

    def translation(self) -> np.ndarray:
        """Tracker origin -> two-finger midpoint, in tracker axes."""
        rotation = self.rotation()
        return (
            self.distance * rotation[:, 0]
            + self.lateral[0] * rotation[:, 1]
            + self.lateral[1] * rotation[:, 2]
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "TrackerToEe":
        lateral = tuple(float(v) for v in (raw.get("lateral") or (0.0, 0.0)))
        return cls(
            approach=(str(raw["approach"]) if raw.get("approach") is not None else None),
            distance=float(raw.get("distance", 0.0)),
            lateral=lateral,  # type: ignore[arg-type]
            spin=float(raw.get("spin", 0.0)),
            measured=bool(raw.get("measured", False)),
        )


def transform_pose_block(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    body: TrackerToEe,
    world: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``^Tracker T_EE`` and the world frame to one pose stream.

    Args:
        positions: ``(N, 3)`` tracker positions as recorded.
        quaternions_wxyz: ``(N, 4)`` tracker orientations, scalar first.
        body: The rig's tracker -> EE transform.
        world: World rotation, or None to leave the recorded frame alone.

    Returns:
        ``(positions, rotation_6d)``, ready to concatenate into the state block.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = quaternions_to_matrices(quaternions_wxyz)
    if len(positions) != len(rotations):
        raise ValueError(f"{len(positions)} positions against {len(rotations)} rotations")

    out_rotations = rotations @ body.rotation()
    out_positions = positions + np.einsum("nij,j->ni", rotations, body.translation())
    if world is not None:
        # Left-multiply only -- see the module docstring.
        out_rotations = world @ out_rotations
        out_positions = out_positions @ world.T
    return out_positions, matrices_to_rotation_6d(out_rotations)
