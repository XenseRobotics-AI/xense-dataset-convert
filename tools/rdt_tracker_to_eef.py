#!/usr/bin/env python
"""Rewrite a ``bi_rdt_gripper`` dataset's tracker poses as EEF poses.

Why this exists
---------------
``{side}_tcp.*`` in a ``bi_rdt_gripper`` dataset is **not** a TCP. It is the raw
HTC Vive tracker pose, straight out of ``*_tracker_LHR-*.hdf5`` — verified
bit-exact against the raw capture (position identical, rotation to 2e-8). The
hdf5→lerobot converter only reorders the quaternion and turns it into the 6D
representation; no rigid transform is applied anywhere.

The TacCap pipeline does apply one, just not in the converter: ``Pico4TrackerReader``
conjugates the world frame and ``robots/taccap_gripper/ee_transform.py``
right-multiplies a CAD-measured ``^Tracker T_EE`` before the HDF5 is ever
written. The RDT capture stack inherits neither, so the offset has to be put
back here.

What it applies
---------------
Body-fixed first, world second, deliberately::

    T_eef = T_tracker @ X                # X = [R | t], in RAW tracker axes
    R_out, t_out = G @ R_eef, G @ t_eef  # world, on by default

Keeping ``X`` on the raw side is what makes the arguments mean what a tape
measure means.

The world step is exact — ``G`` is a convention, not a calibration — so it is on
by default; ``--no-world-up`` leaves the recorded SteamVR frame in place. What it
fixes is gravity only: SteamVR's yaw and origin are set at room-setup time, so
the workspace still faces an arbitrary direction and sits at an arbitrary height,
and those differ between capture sessions.

The world step is a **left-multiply, not a conjugation**. ``G·R·Gᵀ`` re-labels
the body axes as well as the reference frame, which would permute the EEF frame
built above right back out of the convention it was put into — the same trap
``ee_transform.py`` spends a section on as ``APPLY_G_REBASE``. Here the body
frame is physical (``+X`` is the approach, whatever the world is called), so
only the reference frame may change.

The frame that comes out
------------------------
``R`` is built so the EEF frame follows the **same convention TacCap datasets
already carry**: ``+X`` forward along the approach, the other two axes
perpendicular. That is what the rest of the stack assumes — the viewer's
bundled-gripper scene, the pose trajectories, the Doctor's TCP speed check —
so both rigs can be read the same way, and the viewer's RDT-specific axis
cycle (which exists only to paper over the mismatch) can go away.

Choosing the numbers
--------------------
``--approach`` is the tracker axis pointing from the tracker toward the jaws.
Default ``+z``, **measured from the data**: across two datasets, both hands,
four episodes each, the tracker's +Z lies within 45° of vertical-down in 58-81%
of frames while ±X and ±Y never do once.

``--distance`` is tracker origin → two-finger midpoint along that axis, and it is
**provisional**. What the data supports, with no hardware to measure:

* A full 3-parameter pivot calibration (windows where the tip is momentarily
  stationary while the gripper rotates; trimmed refit) puts ``|t|`` at
  **0.18-0.24 m** with the **z component dominant** on every dataset and both
  hands — an independent confirmation of the approach axis.
* But those fits **disagree with each other by 5-15 cm** while each is internally
  precise to 3-9 mm (bootstrap). So each one locks onto a real pivot that is not
  the fingertip — the held object, the wrist — and none of them is the rig
  constant. Precision here is not accuracy.
* The gripper's own URDF is 0.2572 m from base_link to the jaw midpoint. That is
  **not** a lower bound on this number: it would only be one if the tracker sat
  behind the gripper's base on the approach axis, and the pivot fits coming in
  shorter than it suggest the mount is alongside the body instead.

So: somewhere around 0.2-0.3 m, not better than that from this data. A photo of
the rig, the handle CAD, or the RDT2 hardware guide settles it in a minute; until
one of those arrives the output carries ``"provisional": true``.

``--spin`` rotates about the approach axis: it decides which way the jaws open.
Nothing in a pose stream can tell you (the opening direction leaves no trace in
a trajectory), so it is a look-and-see parameter — expect a multiple of 90°.

Neither ``--distance`` nor ``--spin`` can be fitted from the pose data. The
obvious estimator for distance — the fingertip rests on the table, so pick the
offset that makes its height constant — is degenerate here: the approach axis
stays within 43° of vertical, so the offset only shifts the height rather than
trading against the tilt, and the fit scatters from -0.8 m to +1.9 m across
episodes. A tape measure on the rig, or one look at the replay, settles both in
a minute.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - dependency diagnostics
    sys.exit(
        f"{exc}. Install the script dependencies into the interpreter you are "
        f"running ({sys.executable}), e.g. `pip install numpy pyarrow`."
    )

POSE_COLUMNS = ("observation.state", "action")
# [x, y, z, r1..r6, gripper] per arm, left then right.
ARM_STRIDE = 10
ARM_OFFSETS = (0, 10)

# SteamVR/OpenVR (X right, Y up, -Z forward) -> Xense world (X forward, Y left,
# Z up). Identical to `PICO_TO_WORLD_R` in xense-taccap-lerobot: the two devices
# state the same convention, so the same matrix serves. Position (x,y,z) -> (-z,-x,y).
VIVE_TO_WORLD_R = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def rotation_6d_to_matrices(rot6d: np.ndarray) -> np.ndarray:
    """(N,6) -> (N,3,3). First two columns, Gram-Schmidt, third from the cross."""
    a1, a2 = rot6d[:, :3], rot6d[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=2)


def matrices_to_rotation_6d(matrices: np.ndarray) -> np.ndarray:
    """(N,3,3) -> (N,6), the convention hdf52lerobot/geometry.py writes."""
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)


AXES = {
    "+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}


def body_transform(
    approach: np.ndarray, spin_rad: float, distance: float, lateral: tuple[float, float]
) -> np.ndarray:
    """``^Tracker T_EE`` with the EEF's +X on the approach axis.

    The other two axes are free up to a rotation about the approach, which is
    exactly what ``spin`` sets; the seed is picked to be numerically stable
    rather than meaningful, so only the spin is worth reasoning about.
    """
    x_axis = approach / np.linalg.norm(approach)
    seed = np.array([1.0, 0.0, 0.0]) if abs(x_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_seed = np.cross(x_axis, seed)
    y_seed /= np.linalg.norm(y_seed)
    z_seed = np.cross(x_axis, y_seed)
    y_axis = np.cos(spin_rad) * y_seed + np.sin(spin_rad) * z_seed
    z_axis = np.cross(x_axis, y_axis)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform[:3, 3] = distance * x_axis + lateral[0] * y_axis + lateral[1] * z_axis
    return transform


def transform_arm(
    positions: np.ndarray,
    rot6d: np.ndarray,
    body: np.ndarray,
    world: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = rotation_6d_to_matrices(rot6d)
    out_rot = rotations @ body[:3, :3]
    out_pos = positions + np.einsum("nij,j->ni", rotations, body[:3, 3])
    if world is not None:
        # Left-multiply only: re-express in the new world, body frame untouched.
        out_rot = world @ out_rot
        out_pos = out_pos @ world.T
    return out_pos, matrices_to_rotation_6d(out_rot)


def transform_states(states: np.ndarray, body: np.ndarray, world: np.ndarray | None) -> np.ndarray:
    """Transform both arms of an (N, 20) state/action block, gripper untouched."""
    out = states.copy()
    for arm in ARM_OFFSETS:
        positions, rot6d = states[:, arm : arm + 3], states[:, arm + 3 : arm + 9]
        new_positions, new_rot6d = transform_arm(positions, rot6d, body, world)
        out[:, arm : arm + 3] = new_positions
        out[:, arm + 3 : arm + 9] = new_rot6d
    return out


def rewrite_parquet(src: Path, dst: Path, body: np.ndarray, world: np.ndarray | None) -> int:
    table = pq.read_table(src)
    rows = table.num_rows
    columns = {}
    for name in POSE_COLUMNS:
        if name not in table.column_names:
            continue
        column = table.column(name).combine_chunks()
        values = np.asarray(column.to_pylist(), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != ARM_STRIDE * len(ARM_OFFSETS):
            raise SystemExit(f"{name} is {values.shape}, expected (N, 20) — is this bi_rdt_gripper?")
        transformed = transform_states(values, body, world)
        # Rebuild as the same list<float> type so the column round-trips exactly.
        columns[name] = pa.array(transformed.astype(np.float32).tolist(), type=column.type)

    arrays = [columns.get(name, table.column(name)) for name in table.column_names]
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=table.schema), dst)
    return rows


def update_stats(path: Path, body: np.ndarray, world: np.ndarray | None, data_files: list[Path]) -> None:
    """Recompute min/max/mean/std for the pose columns; leave everything else."""
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    stacked = {name: [] for name in POSE_COLUMNS}
    for data_file in data_files:
        table = pq.read_table(data_file, columns=[c for c in POSE_COLUMNS if c in pq.read_schema(data_file).names])
        for name in table.column_names:
            stacked[name].append(np.asarray(table.column(name).combine_chunks().to_pylist(), dtype=np.float64))
    for name, blocks in stacked.items():
        if not blocks or name not in stats:
            continue
        values = np.concatenate(blocks, axis=0)
        entry = stats[name]
        entry["min"] = values.min(axis=0).tolist()
        entry["max"] = values.max(axis=0).tolist()
        entry["mean"] = values.mean(axis=0).tolist()
        entry["std"] = values.std(axis=0).tolist()
    path.write_text(json.dumps(stats, indent=4))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # same filesystem: instant, and realpath stays inside the dataset
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="source lerobot dataset directory")
    parser.add_argument("--dst", required=True, type=Path, help="destination directory (created)")
    parser.add_argument(
        "--approach",
        choices=sorted(AXES),
        default="+z",
        help="tracker axis pointing at the jaws; becomes the EEF's +X. Default: +z (measured)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.22,
        metavar="M",
        help="tracker origin -> two-finger midpoint along the approach axis (m). "
        "Provisional; the data only bounds it to roughly 0.2-0.3 m. Default: 0.22",
    )
    parser.add_argument(
        "--lateral",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("A", "B"),
        help="offsets perpendicular to the approach, along the EEF's Y and Z (m). Default: 0 0",
    )
    parser.add_argument(
        "--spin",
        type=float,
        default=0.0,
        metavar="DEG",
        help="rotation about the approach axis, degrees — picks which way the jaws open. Default: 0",
    )
    parser.add_argument(
        "--world-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="convert SteamVR's Y-up world to the Xense X-forward/Y-left/Z-up world (default: on)",
    )
    parser.add_argument(
        "--measured",
        action="store_true",
        help="the offset/spin came from the rig, not from a guess — drops the provisional marker",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing destination")
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"{src} has no meta/info.json — not a lerobot dataset.")
    info = json.loads((src / "meta" / "info.json").read_text())
    robot_type = info.get("robot_type")
    if robot_type != "bi_rdt_gripper":
        print(f"warning: robot_type is {robot_type!r}, not 'bi_rdt_gripper' — the "
              f"tracker-pose finding was only verified for RDT captures.", file=sys.stderr)
    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} exists; pass --force to overwrite.")
        shutil.rmtree(dst)

    if not args.measured:
        print(f"warning: --distance {args.distance:.3f} m and --spin {args.spin:g}° are PROVISIONAL "
              f"(see the module docstring: the pose data bounds the offset to roughly 0.2-0.3 m and "
              f"says nothing at all about the spin). Pass --measured once they come from the rig.",
              file=sys.stderr)
    approach = AXES[args.approach]
    body = body_transform(approach, np.radians(args.spin), args.distance, tuple(args.lateral))
    world = VIVE_TO_WORLD_R if args.world_up else None

    print(f"src        {src}")
    print(f"dst        {dst}")
    print(f"approach   tracker {args.approach} -> becomes the EEF's +X")
    print(f"offset     {args.distance} m along it, lateral {list(args.lateral)} m")
    print(f"spin       {args.spin}° about the approach axis")
    print(f"world      {'SteamVR Y-up -> Xense Z-up' if args.world_up else 'left as recorded (SteamVR Y-up)'}")

    # meta and videos ride along untouched; only the pose columns change.
    shutil.copytree(src / "meta", dst / "meta")
    total = 0
    data_files = []
    for parquet in sorted((src / "data").rglob("*.parquet")):
        out = dst / parquet.relative_to(src)
        total += rewrite_parquet(parquet, out, body, world)
        data_files.append(out)
    for video in sorted((src / "videos").rglob("*")):
        if video.is_file():
            link_or_copy(video, dst / video.relative_to(src))

    update_stats(dst / "meta" / "stats.json", body, world, data_files)
    (dst / "meta" / "rdt_eef_transform.json").write_text(
        json.dumps(
            {
                "format": "rdt_tracker_to_eef/1",
                "source": str(src),
                "approach_axis_in_tracker": args.approach,
                "tracker_to_eef_distance_m": args.distance,
                "tracker_to_eef_lateral_m": list(args.lateral),
                "tracker_to_eef_spin_deg": args.spin,
                "eef_frame": "+X forward along the approach (same convention as bi_taccap_gripper)",
                "world_up_fix": bool(args.world_up),
                "provisional": not args.measured,
                "note": "Recorded {side}_tcp was the raw Vive tracker pose; this copy has the "
                        "body-fixed tracker->EEF transform applied. The world rotation is exact; "
                        "the distance and spin are provisional unless provisional is false.",
            },
            indent=2,
        )
    )
    print(f"rewrote    {total} rows across {len(data_files)} parquet file(s)")


if __name__ == "__main__":
    main()
