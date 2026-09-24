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

The shape is the one xense-taccap-lerobot's ``manifest_epochs`` reads, so a
consumer written against TacCap datasets reads these without branching:
``{robot_type, epochs[]}``, each epoch ``{from_episode, to_episode, robot_id,
role, units[]}``, each unit ``{side, gripper_sn, tactile_sensors[]}``, each
sensor ``{finger, observation_key, serial}``. ``to_episode`` is exclusive.

Two of those are null here rather than absent, which is the honest way to say
"this rig has no such thing": ``gripper_sn`` (the recordings name the jaw
encoder by its calibration range, not a firmware serial) and ``role`` (a
handheld UMI rig has no leader or follower).

Three things this rig records that TacCap's schema has no place for, kept
alongside rather than squeezed in: the jaw encoder's calibration range, the
tracker and wrist-camera serials, and the fixed side camera. A reader that
knows only the TacCap shape ignores them; one that knows this rig has them by
name.

``robot_id`` appears twice, as the schema has it: per epoch, and once at the
top for the dataset. A dataset gathering several stations has no single one, so
the top-level value is ``<robot_type>_unknown`` -- stated rather than omitted,
because a reader can act on a value and cannot act on a missing key.

The station is named by ``robot_id`` and nothing else. An earlier version also
wrote it as ``source``, which was a duplicate with a worse name: ``robot_id``
is what ``manifest_robot_ids`` and ``check_dataset_station`` already read, and
a second key meaning the same thing is a second key that can disagree. A
merge-provenance field, when merging grows one, is a different fact and takes a
different name.

`tools/rdt_convert.py` writes this after each conversion. To rewrite the
manifest of a dataset converted earlier, from the episodes it actually holds:

    python tools/rdt_hardware.py --dataset ~/.cache/huggingface/lerobot/TacVerse-RDT-Test/xtac-rdt-umi-*
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


_EPISODE_NAME = re.compile(r"^ep_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")

# The stations record in local time, China Standard Time. Checked against the
# data rather than assumed: ep_20260804_131436's first sample is at 13:14:37
# +08:00.
RECORDER_UTC_OFFSET = "+08:00"


def recorded_at(episode: Path) -> str | None:
    """When an episode was recorded, from its directory name.

    ``ep_20260810_120057`` is the recorder's own naming, so the time is in the
    dataset rather than inferred from a file's mtime. Written in the form
    TacCap uses, ``2026-08-10T12:00:57+08:00``. It feeds TacFlow's strongest
    tier of capture-date evidence (`registry.capture_dates`, which ranks the
    manifest above a source name above a directory name), and the viewer's.
    """
    match = _EPISODE_NAME.match(episode.name)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}{RECORDER_UTC_OFFSET}"


def _epoch_body(hardware: dict[str, Any], robot_prefix: str) -> dict[str, Any]:
    station = hardware["station"]
    units = []
    for side in ("left", "right"):
        found = hardware["units"][side]
        units.append(
            {
                # The five keys TacCap's schema defines, in its order, first.
                "side": side,
                "gripper_sn": None,
                "tactile_sensors": [
                    {
                        "finger": finger,
                        "observation_key": f"{side}_tactile_{finger}",
                        "serial": found["tactile"].get(finger),
                    }
                    for finger in ("left", "right")
                ],
                # And this rig's own, which TacCap has no place for.
                "unit_id": found.get("unit_id"),
                "gripper_calibration": found.get("gripper_calibration"),
                "tracker_serial": found.get("tracker"),
                "wrist_camera_serial": found.get("wrist_camera"),
            }
        )
    return {
        "robot_id": f"{robot_prefix}_{station}",
        # Null, not absent: this rig has no leader/follower pair to name.
        "role": None,
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
        body["recorded_at"] = recorded_at(Path(episode))
        if epochs and _same_epoch(epochs[-1], body):
            epochs[-1]["to_episode"] = index + 1
            continue
        epochs.append({"from_episode": index, "to_episode": index + 1, **body})
    return {
        "robot_type": robot_type,
        "robot_id": dataset_robot_id(epochs, robot_type),
        "epochs": epochs,
    }


def dataset_robot_id(epochs: Sequence[dict[str, Any]], robot_type: str) -> str:
    """The station for the dataset as a whole.

    The schema states it twice -- once here and once per epoch -- because at
    this level it is an invariant rather than a fact: nothing may resume this
    dataset from another station. A TacCap dataset is one station, so the two
    agree by construction.

    An RDT dataset is not. It gathers every station that recorded the task that
    day, so no single station names it, and the invariant does not hold. That is
    said with ``<robot_type>_unknown`` rather than by leaving the key out: a
    reader can tell "many stations, do not resume here" from a value, and cannot
    tell it from an absence, which also means "written by something that forgot".
    """
    stations = {epoch.get("robot_id") for epoch in epochs}
    if len(stations) == 1:
        only = stations.pop()
        if only:
            return only
    return f"{robot_type}_unknown"


def _same_epoch(epoch: dict[str, Any], body: dict[str, Any]) -> bool:
    """Whether an episode extends the last epoch rather than opening one.

    The same hardware on the same day. An epoch's ``recorded_at`` is when its
    first episode was recorded, as in TacCap, where an epoch is a recording
    session; comparing the full timestamp would make every episode an epoch.
    The day still has to match: coalescing two days recorded on an unchanged
    rig would drop one day's ``recorded_at`` -- silently demoting TacFlow's
    capture-date evidence from the manifest to a guess at the source name
    (``registry.capture_dates``). An epoch too many costs a row, an epoch too
    few costs a fact.
    """
    if _day(epoch.get("recorded_at")) != _day(body.get("recorded_at")):
        return False
    return all(
        epoch.get(key) == value for key, value in body.items()
        if key not in ("from_episode", "to_episode", "recorded_at")
    )


def _day(timestamp: str | None) -> str | None:
    return timestamp[:10] if timestamp else None


def converted_episodes(dataset_root: Path) -> list[Path]:
    """The recordings a dataset holds, in episode order, from ``meta/sources.json``.

    Not the list the conversion was given: recordings that fail to convert are
    skipped, and every episode after one moves down an index. A manifest built
    from the inputs would have its ranges run past the dataset's end and name
    the wrong station for every episode after the first skip.
    """
    sources = json.loads((Path(dataset_root) / "meta" / "sources.json").read_text(encoding="utf-8"))
    ordered = sorted(sources["episodes"], key=lambda entry: entry["episode_index"])
    indices = [entry["episode_index"] for entry in ordered]
    if indices != list(range(len(indices))):
        raise ValueError(f"{dataset_root}: meta/sources.json episode indices are not 0..{len(indices) - 1}")
    return [Path(entry["source_dir"]) for entry in ordered]


def write_hardware(dataset_root: Path, robot_type: str) -> Path:
    """Write ``meta/hardware.json`` for the episodes the dataset actually holds."""
    path = Path(dataset_root) / "meta" / "hardware.json"
    document = hardware_document(converted_episodes(dataset_root), robot_type=robot_type)
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dataset",
        action="append",
        type=Path,
        help="Converted dataset whose meta/hardware.json to rewrite from its meta/sources.json. Repeatable.",
    )
    mode.add_argument("--input", action="append", help="Recording directory or station. Repeatable.")
    parser.add_argument("--robot-type", default="bi_rdt_gripper")
    parser.add_argument("--output", help="With --input: write here (default: stdout).")
    args = parser.parse_args(argv)

    if args.dataset:
        for dataset in args.dataset:
            path = write_hardware(dataset, args.robot_type)
            document = json.loads(path.read_text(encoding="utf-8"))
            episodes = document["epochs"][-1]["to_episode"] if document["epochs"] else 0
            print(f"{dataset.name}: {len(document['epochs'])} epoch(s), {episodes} episode(s) -> {path}")
        return 0

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
