#!/usr/bin/env python3
"""Build ``meta/hardware.json`` for a converted TacVerse-RDT dataset.

TacVerse datasets record which physical hardware produced which episodes, as a
list of epochs over episode ranges: the machine, and per side the gripper and
the two tactile sensors mapped to their observation keys. The RDT recordings
carry the same information in their file names -- every camera, tactile sensor
and tracker is named by its serial, and the jaw encoder by its calibration
bounds -- so nothing has to be asked for; it only has to be read.

An epoch ends wherever the hardware changes. That is usually a change of
machine, since a dataset merges every station that recorded the task that day,
but it also catches a sensor swapped mid-session: rig 036's right-hand
``tactile_left`` changed from OG001652 to OG001868 partway through August.

Two fields TacVerse carries cannot be filled: the gripper serial (the
recordings name the jaw encoder by its calibration range instead, kept here as
``gripper_calibration``) and ``role`` (a handheld UMI rig has no leader or
follower). Camera and tracker serials are recorded, which TacVerse's own
schema does not carry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

SIDECAR = ".timestamp.hdf5"

# `<unit>` is the station's own label for the device, e.g. 036L for rig 036's
# left hand; the serial is what identifies the hardware itself.
PATTERNS = {
    "view_camera": re.compile(r"^head_(?P<unit>\w+?)_mvs_mono_(?P<serial>[A-Za-z0-9]+)_\w+\.mkv$"),
    "wrist_camera": re.compile(
        r"^(?P<side>left|right)_umi_(?P<unit>\w+?)_mvs_mono_(?P<serial>[A-Za-z0-9]+)_\w+\.mkv$"
    ),
    "tactile": re.compile(
        r"^(?P<side>left|right)_umi_(?P<unit>\w+?)_tactile_(?P<finger>left|right)_"
        r"(?P<serial>[A-Za-z0-9]+)_\w+\.mkv$"
    ),
    "tracker": re.compile(
        r"^(?P<side>left|right)_umi_(?P<unit>\w+?)_tracker_(?P<serial>[\w-]+)\.hdf5$"
    ),
    "gripper": re.compile(
        r"^(?P<side>left|right)_umi_(?P<unit>\w+?)_openv2_(?P<lo>[\d.]+)_(?P<hi>[\d.]+)\.hdf5$"
    ),
}


def read_episode_hardware(episode: Path) -> dict[str, Any]:
    """The hardware one recording directory names, from its file names alone."""
    unit: dict[str, dict[str, Any]] = {
        "left": {"tactile": {}},
        "right": {"tactile": {}},
    }
    view_camera: dict[str, str] = {}
    for entry in sorted(episode.iterdir()):
        name = entry.name
        if name.endswith(SIDECAR):
            continue
        for kind, pattern in PATTERNS.items():
            found = pattern.match(name)
            if not found:
                continue
            groups = found.groupdict()
            if kind == "view_camera":
                view_camera = {"unit_id": groups["unit"], "serial": groups["serial"]}
            elif kind == "tactile":
                unit[groups["side"]]["tactile"][groups["finger"]] = groups["serial"]
                unit[groups["side"]]["unit_id"] = groups["unit"]
            elif kind == "gripper":
                unit[groups["side"]]["gripper_calibration"] = [
                    float(groups["lo"]),
                    float(groups["hi"]),
                ]
                unit[groups["side"]]["unit_id"] = groups["unit"]
            else:
                unit[groups["side"]][kind] = groups["serial"]
                unit[groups["side"]]["unit_id"] = groups["unit"]
            break
    return {"station": episode.parent.name, "view_camera": view_camera, "units": unit}


def _epoch_body(hardware: dict[str, Any], robot_prefix: str) -> dict[str, Any]:
    station = hardware["station"]
    units = []
    for side in ("left", "right"):
        found = hardware["units"][side]
        units.append(
            {
                "side": side,
                "unit_id": found.get("unit_id"),
                "gripper_sn": None,
                "gripper_calibration": found.get("gripper_calibration"),
                "tracker_serial": found.get("tracker"),
                "wrist_camera_serial": found.get("wrist_camera"),
                "tactile_sensors": [
                    {
                        "finger": finger,
                        "observation_key": f"{side}_tactile_{finger}",
                        "serial": found["tactile"].get(finger),
                    }
                    for finger in ("left", "right")
                ],
            }
        )
    return {
        "source": station,
        "robot_id": f"{robot_prefix}_{station}",
        "units": units,
        "cameras": [
            {
                "observation_key": "observation.images.side",
                "serial": hardware["view_camera"].get("serial"),
                "unit_id": hardware["view_camera"].get("unit_id"),
            },
            {"observation_key": "observation.images.left_wrist", "serial": units[0]["wrist_camera_serial"]},
            {"observation_key": "observation.images.right_wrist", "serial": units[1]["wrist_camera_serial"]},
        ],
    }


def hardware_document(
    episodes: Sequence[Path],
    robot_type: str = "bi_rdt_gripper",
    robot_prefix: str = "rdt_umi",
) -> dict[str, Any]:
    """``{robot_type, epochs}`` for the episodes, in the order they were written.

    Consecutive episodes sharing the same hardware collapse into one epoch, so
    a dataset merging thirty stations gets thirty ranges rather than eleven
    thousand.
    """
    epochs: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        body = _epoch_body(read_episode_hardware(Path(episode)), robot_prefix)
        if epochs and _same_hardware(epochs[-1], body):
            epochs[-1]["to_episode"] = index + 1
            continue
        epochs.append({"from_episode": index, "to_episode": index + 1, **body})
    return {"robot_type": robot_type, "epochs": epochs}


def _same_hardware(epoch: dict[str, Any], body: dict[str, Any]) -> bool:
    keys = ("source", "robot_id", "units", "cameras")
    return all(epoch.get(key) == body[key] for key in keys)


def write_hardware(dataset_root: Path, episodes: Sequence[Path], robot_type: str) -> Path:
    """Write ``meta/hardware.json`` next to the dataset's other metadata."""
    path = Path(dataset_root) / "meta" / "hardware.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = hardware_document(episodes, robot_type=robot_type)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _episode_dirs(inputs: Iterable[str]) -> list[Path]:
    episodes: list[Path] = []
    for entry in inputs:
        path = Path(entry).expanduser().resolve()
        episodes.extend(
            [path] if path.name.startswith("ep_") else sorted(p for p in path.iterdir() if p.is_dir())
        )
    return episodes


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", action="append", required=True, help="Recording directory or station. Repeatable.")
    parser.add_argument("--robot-type", default="bi_rdt_gripper")
    parser.add_argument("--output", help="Write here (default: stdout).")
    args = parser.parse_args(argv)

    document = hardware_document(_episode_dirs(args.input), robot_type=args.robot_type)
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).expanduser().write_text(text, encoding="utf-8")
        print(f"{len(document['epochs'])} epoch(s) -> {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
