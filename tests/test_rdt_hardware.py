"""The RDT hardware manifest, in the shape TacCap datasets already use.

A consumer written against one family should read the other without branching,
so the check that matters is not "does it have the keys we chose" but "does the
fork's own reader make sense of it". `manifest_epochs` is that reader, and it is
used here when the fork is on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.rdt_hardware import hardware_document, write_hardware  # noqa: E402

FORK_SOURCE = Path.home() / "xense-taccap-lerobot" / "src"

# One station's file set, named the way the recordings are: every serial is in
# the name, which is why nothing has to be asked for.
NAMES = (
    "head_035H_mvs_mono_DA6135149_libx264.mkv",
    "left_umi_035L_mvs_mono_DA6022614_libx264.mkv",
    "left_umi_035L_openv2_131.11_158.53.hdf5",
    "left_umi_035L_tactile_left_OG001524_libsvtav1.mkv",
    "left_umi_035L_tactile_right_OG001525_libsvtav1.mkv",
    "left_umi_035L_tracker_LHR-D70F7813.hdf5",
    "right_umi_035R_mvs_mono_DA6022627_libx264.mkv",
    "right_umi_035R_openv2_247.90_275.38.hdf5",
    "right_umi_035R_tactile_left_OG001526_libsvtav1.mkv",
    "right_umi_035R_tactile_right_OG001527_libsvtav1.mkv",
    "right_umi_035R_tracker_LHR-FCE8E8E8.hdf5",
)


def episode(root: Path, station: str, name: str, **swaps: str) -> Path:
    """An episode directory whose file names carry the hardware."""
    directory = root / station / name
    directory.mkdir(parents=True)
    for filename in NAMES:
        for old, new in swaps.items():
            filename = filename.replace(old, new)
        (directory / filename.replace("035", station)).touch()
    return directory


def test_the_manifest_is_the_shape_the_fork_reads(tmp_path: Path):
    document = hardware_document([episode(tmp_path, "035", "ep_20260804_090000")])

    assert sorted(document) == ["epochs", "robot_id", "robot_type"]
    assert document["robot_type"] == "bi_rdt_gripper"
    # One station, so the dataset-level invariant and the epoch agree.
    assert document["robot_id"] == "rdt_umi_035"
    epoch = document["epochs"][0]
    for key in ("from_episode", "to_episode", "robot_id", "role", "units"):
        assert key in epoch, key
    assert epoch["from_episode"] == 0 and epoch["to_episode"] == 1
    assert epoch["robot_id"] == "rdt_umi_035"
    # Null rather than absent: a handheld rig has no leader/follower to name.
    assert epoch["role"] is None

    unit = epoch["units"][0]
    assert unit["side"] == "left"
    # Null for the same reason: the jaw encoder has no firmware serial.
    assert unit["gripper_sn"] is None
    sensor = unit["tactile_sensors"][0]
    assert sorted(sensor) == ["finger", "observation_key", "serial"]
    assert sensor["observation_key"] == "left_tactile_left"
    assert sensor["serial"] == "OG001524"

    # What this rig has and TacCap's schema does not; a fork reader ignores them.
    assert unit["gripper_calibration"] == [131.11, 158.53]
    assert unit["tracker_serial"] == "LHR-D70F7813"
    assert unit["wrist_camera_serial"] == "DA6022614"
    assert epoch["cameras"][0]["observation_key"] == "observation.images.side"


def test_the_fork_reader_accepts_it(tmp_path: Path):
    """Checked against `manifest_epochs` itself, skipped where the fork is not."""
    document = hardware_document([episode(tmp_path, "035", "ep_20260804_090000")])
    if not (FORK_SOURCE / "lerobot" / "robots" / "taccap_gripper" / "common.py").exists():
        pytest.skip(f"xense-taccap-lerobot not checked out at {FORK_SOURCE}")
    # `lerobot` in this environment is the installed upstream, which has no
    # taccap_gripper, so the fork's tree has to come first and the already
    # imported upstream has to go.
    import subprocess

    probe = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "from lerobot.robots.taccap_gripper.common import manifest_epochs;"
        "print(json.dumps(manifest_epochs(json.loads(sys.stdin.read()))))"
    ) % str(FORK_SOURCE)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        input=json.dumps(document),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot import the fork: {result.stderr.strip().splitlines()[-1:]}")
    epochs = json.loads(result.stdout)

    assert len(epochs) == 1
    # Read as-is, not lifted through the pre-epoch fallback, which would have
    # given to_episode None.
    assert epochs[0]["to_episode"] == 1
    assert epochs[0]["units"][0]["tactile_sensors"][0]["serial"] == "OG001524"


def test_an_epoch_ends_where_the_hardware_changes(tmp_path: Path):
    """A sensor swapped mid-session opens a new epoch, as on rig 036 in August."""
    first = episode(tmp_path, "036", "ep_20260804_090000")
    second = episode(tmp_path, "036", "ep_20260804_100000")
    third = episode(tmp_path, "036", "ep_20260804_110000", OG001526="OG001868")

    document = hardware_document([first, second, third])
    epochs = document["epochs"]
    assert len(epochs) == 2, "the swap should split, and the two before it should not"
    assert (epochs[0]["from_episode"], epochs[0]["to_episode"]) == (0, 2)
    assert (epochs[1]["from_episode"], epochs[1]["to_episode"]) == (2, 3)

    def right_tactile_left(epoch):
        right = next(u for u in epoch["units"] if u["side"] == "right")
        return next(s for s in right["tactile_sensors"] if s["finger"] == "left")["serial"]

    assert right_tactile_left(epochs[0]) == "OG001526"
    assert right_tactile_left(epochs[1]) == "OG001868"


def test_every_episode_falls_in_exactly_one_epoch(tmp_path: Path):
    episodes = [
        episode(tmp_path, station, f"ep_2026080{n}_090000")
        for n, station in enumerate(("035", "035", "036", "035"), start=1)
    ]
    document = hardware_document(episodes)

    covered = [
        index
        for epoch in document["epochs"]
        for index in range(epoch["from_episode"], epoch["to_episode"])
    ]
    assert covered == list(range(len(episodes)))


def test_a_dataset_spanning_stations_says_so_rather_than_omitting(tmp_path: Path):
    """No single station names it, and that is a value, not a missing key."""
    document = hardware_document(
        [
            episode(tmp_path, "035", "ep_20260804_090000"),
            episode(tmp_path, "036", "ep_20260804_100000"),
        ]
    )
    assert document["robot_id"] == "bi_rdt_gripper_unknown"
    # The epochs still name their own stations; only the invariant is absent.
    assert [epoch["robot_id"] for epoch in document["epochs"]] == [
        "rdt_umi_035",
        "rdt_umi_036",
    ]


def test_recorded_at_comes_from_the_episode_name(tmp_path: Path):
    """The recorder's own naming, so the time is in the dataset, not inferred.

    It also keeps two days on one rig in separate epochs: coalescing them would
    drop a `recorded_at`, which is the strongest capture-date evidence TacFlow
    has (`registry.capture_dates` ranks it above a source name).
    """
    document = hardware_document(
        [
            episode(tmp_path, "035", "ep_20260804_090000"),
            episode(tmp_path, "035", "ep_20260805_131436"),
        ]
    )
    assert [epoch["recorded_at"] for epoch in document["epochs"]] == [
        "2026-08-04T09:00:00+08:00",
        "2026-08-05T13:14:36+08:00",
    ]
    assert len(document["epochs"]) == 2, "same rig, different days, still two epochs"
    # And the dataset is still one station.
    assert document["robot_id"] == "rdt_umi_035"


def test_one_day_on_one_rig_is_one_epoch_from_its_first_recording(tmp_path: Path):
    """A timestamp per episode must not turn every episode into an epoch."""
    document = hardware_document(
        [
            episode(tmp_path, "035", "ep_20260804_090000"),
            episode(tmp_path, "035", "ep_20260804_093015"),
            episode(tmp_path, "035", "ep_20260804_150000"),
        ]
    )
    assert len(document["epochs"]) == 1
    assert document["epochs"][0]["to_episode"] == 3
    assert document["epochs"][0]["recorded_at"] == "2026-08-04T09:00:00+08:00"


def test_written_for_the_episodes_the_dataset_holds(tmp_path: Path):
    """A recording skipped during conversion takes no index in the manifest.

    Built from the conversion's inputs instead, every range after the skip
    runs one past the dataset and names the wrong station at the boundary.
    """
    kept_035 = episode(tmp_path / "raw", "035", "ep_20260804_090000")
    episode(tmp_path / "raw", "035", "ep_20260804_091500")  # skipped
    kept_036 = episode(tmp_path / "raw", "036", "ep_20260804_100000")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    sources = {
        # Listed out of order on purpose: episode_index decides, not position.
        "episodes": [
            {"episode_index": 1, "source_dir": str(kept_036)},
            {"episode_index": 0, "source_dir": str(kept_035)},
        ],
        "skipped": [{"source_dir": str(tmp_path / "raw" / "035" / "ep_20260804_091500")}],
    }
    (dataset / "meta" / "sources.json").write_text(json.dumps(sources), encoding="utf-8")

    document = json.loads(write_hardware(dataset, "bi_rdt_gripper").read_text(encoding="utf-8"))

    assert [(e["from_episode"], e["to_episode"], e["robot_id"]) for e in document["epochs"]] == [
        (0, 1, "rdt_umi_035"),
        (1, 2, "rdt_umi_036"),
    ]
