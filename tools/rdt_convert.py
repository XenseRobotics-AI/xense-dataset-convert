#!/usr/bin/env python3
"""Convert the TacVerse-RDT recordings into one LeRobot dataset per task.

TacVerse's convention is one dataset per task, named for the task and the day
it was captured. The RDT recordings are filed by capture station instead, and
what a session's task was lives in the capture team's schedule: a task code per
half-day, spelled out in `configs/tacverse_rdt_catalogue.yaml`.

So a dataset here is one (task, date): every station that recorded that task
that day, merged, with `meta/hardware.json` recording which station produced
which episodes. Episodes are ordered station by station so each one occupies a
single contiguous range.

    python tools/rdt_convert.py --list
    python tools/rdt_convert.py --only xtac-rdt-umi-turn-rotary-knob-0804 --limit 40
    python tools/rdt_convert.py --jobs 5

A half-day that lists several task codes is skipped, and counted in the
summary: the schedule says the session covered two or three tasks but not
which recording is which, so those episodes cannot be filed by task until the
capture team supplies a per-episode list. `--list` reports how many that is.

Converting is resumable twice over. A run stopped with Ctrl-C finishes the
episode it is on, writes `meta/.checkpoint.json` and exits; running it again
carries on from there, and the result is identical to never having stopped. A
dataset whose `meta/info.json` already exists is
left alone unless `--overwrite` is given, and one whose directory exists
*without* that file is a run that was interrupted, so it is converted again
from scratch. Only one driver may work on an output tree at a time -- a second
one would race the first over the same directories -- which a lock file
enforces.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rdt_hardware import write_hardware  # noqa: E402
from tools.rdt_task_map import read_schedule  # noqa: E402

CODE_SEPARATORS = re.compile(r"[、,，/]")
EPISODE_NAME = re.compile(r"^ep_(\d{8})_(\d{6})$")
DEFAULT_RAW = Path("/data/data4/umi_data")
DEFAULT_SCHEDULE = Path.home() / "Documents/TacVerse-RDT-Rules/采集任务时间安排表.xlsx"
REPO = Path(__file__).resolve().parent.parent


@dataclass
class Group:
    """One (task, date): the dataset it becomes and the episodes it holds."""

    code: str
    day: datetime.date
    name: str
    task: str
    episodes: list[Path]

    @property
    def stations(self) -> list[str]:
        return sorted({episode.parent.name for episode in self.episodes})


def load_catalogue(path: Path) -> dict[str, dict]:
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["tasks"]


def build_groups(
    raw_root: Path,
    schedule: dict[datetime.date, tuple[str, str]],
    catalogue: dict[str, dict],
    prefix: str,
) -> tuple[list[Group], dict[str, int]]:
    """Group every recording by (task, date), and count what cannot be grouped."""
    groups: dict[tuple[str, datetime.date], list[Path]] = {}
    skipped = {"multi_code": 0, "unscheduled": 0, "unknown_code": 0, "bad_name": 0}

    for station in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for episode in sorted(station.iterdir()):
            found = EPISODE_NAME.match(episode.name)
            if not found or not episode.is_dir():
                skipped["bad_name"] += 1
                continue
            stamp = datetime.datetime.strptime(
                f"{found.group(1)}{found.group(2)}", "%Y%m%d%H%M%S"
            )
            half_days = schedule.get(stamp.date())
            if not half_days:
                skipped["unscheduled"] += 1
                continue
            codes = [
                code.strip()
                for code in CODE_SEPARATORS.split(half_days[0 if stamp.hour < 12 else 1])
                if code.strip()
            ]
            if len(codes) != 1:
                skipped["multi_code" if codes else "unscheduled"] += 1
                continue
            if codes[0] not in catalogue:
                skipped["unknown_code"] += 1
                continue
            groups.setdefault((codes[0], stamp.date()), []).append(episode)

    built = [
        Group(
            code=code,
            day=day,
            name=f"{prefix}-{catalogue[code]['slug']}-{day.strftime('%m%d')}",
            task=catalogue[code]["en"],
            # Station by station, so hardware.json gets one range per station.
            episodes=sorted(episodes, key=lambda p: (p.parent.name, p.name)),
        )
        for (code, day), episodes in sorted(groups.items())
    ]
    return built, skipped


def convert_group(group: Group, output_root: Path, config_path: Path, workers: int, overwrite: bool) -> dict:
    """Convert one group into one dataset, then write its hardware epochs."""
    from hdf52lerobot.config import ConverterConfig
    from hdf52lerobot.converter import Hdf5ToLeRobotConverter

    destination = output_root / group.name
    finished = (destination / "meta" / "info.json").exists()
    if finished and not overwrite:
        return {"dataset": group.name, "status": "skipped", "reason": "already converted"}
    # A directory holding a checkpoint is an interrupted run, and the converter
    # carries on from it. Anything else without info.json is unusable and gets
    # replaced -- the converter would otherwise refuse to write into it.
    from hdf52lerobot.checkpoint import Checkpoint

    resumable = Checkpoint.load(destination) is not None
    replace = overwrite or (destination.exists() and not resumable)

    config = ConverterConfig.from_yaml(config_path).with_overrides(task=group.task, workers=workers)
    started = time.time()
    try:
        Hdf5ToLeRobotConverter(
            inputs=group.episodes,
            output_root=destination,
            config=config,
            overwrite=replace,
            # One damaged recording out of thousands should cost that recording,
            # not the dataset; the skipped ones are listed in meta/sources.json.
            skip_failed_episodes=True,
        ).convert()
    except KeyboardInterrupt as stop:
        # The checkpoint is already written; say where, and leave the dataset
        # unfinished rather than pretending either way.
        return {
            "dataset": group.name,
            "status": "interrupted",
            "detail": str(stop),
            "seconds": round(time.time() - started, 1),
        }
    write_hardware(destination, group.episodes, config.robot_type)

    info = json.loads((destination / "meta" / "info.json").read_text(encoding="utf-8"))
    sources = json.loads((destination / "meta" / "sources.json").read_text(encoding="utf-8"))
    return {
        "dataset": group.name,
        "status": "converted",
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "skipped": len(sources.get("skipped", [])),
        "stations": len(group.stations),
        "seconds": round(time.time() - started, 1),
    }


def run_in_parallel(groups: list[Group], args: argparse.Namespace) -> int:
    """Run one child process per dataset, `--jobs` of them at a time.

    A dataset is a single sequential write, so the way to use a machine with
    more cores than one conversion can occupy is to convert several datasets at
    once. Each child re-derives its own group from the schedule, which keeps
    this driver free of any state to hand over.
    """
    pending = list(groups)
    running: list[tuple[subprocess.Popen, Group]] = []
    failures = 0
    log_dir = args.output / ".convert-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or running:
        while pending and len(running) < args.jobs:
            group = pending.pop(0)
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--raw", str(args.raw), "--schedule", str(args.schedule),
                "--catalogue", str(args.catalogue), "--config", str(args.config),
                "--output", str(args.output), "--prefix", args.prefix,
                "--workers", str(args.workers), "--only", group.name,
            ]
            if args.overwrite:
                command.append("--overwrite")
            if args.limit:
                command += ["--limit", str(args.limit)]
            # Append: a resumed run should add to the record, not erase it.
            log = (log_dir / f"{group.name}.log").open("a", encoding="utf-8")
            print(f"[start] {group.name}  {len(group.episodes)} episodes", flush=True)
            running.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), group))

        time.sleep(2)
        for process, group in list(running):
            if process.poll() is None:
                continue
            running.remove((process, group))
            if process.returncode == 0:
                print(f"[done ] {group.name}", flush=True)
            else:
                failures += 1
                print(
                    f"[FAIL ] {group.name} (exit {process.returncode}) "
                    f"-> {log_dir / (group.name + '.log')}",
                    flush=True,
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Root holding <station>/ep_* directories.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--catalogue", type=Path, default=REPO / "configs/tacverse_rdt_catalogue.yaml")
    parser.add_argument("--config", type=Path, default=REPO / "configs/tacverse_rdt.yaml")
    parser.add_argument("--output", type=Path, default=Path("/archive/lerobot/TacVerse/raw"))
    parser.add_argument("--prefix", default="xtac-rdt-umi")
    parser.add_argument("--only", help="Convert just this dataset name.")
    parser.add_argument("--limit", type=int, help="Take only the first N episodes of each group (for a trial run).")
    parser.add_argument("--jobs", type=int, default=1, help="Datasets converted concurrently.")
    parser.add_argument("--workers", type=int, default=4, help="Encoder threads per conversion.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="Show what would be converted and stop.")
    args = parser.parse_args(argv)

    _, schedule = read_schedule(args.schedule)
    catalogue = load_catalogue(args.catalogue)
    groups, skipped = build_groups(args.raw, schedule, catalogue, args.prefix)

    if args.only:
        groups = [group for group in groups if group.name == args.only]
        if not groups:
            print(f"No dataset named {args.only!r}", file=sys.stderr)
            return 2
    if args.limit:
        groups = [
            Group(g.code, g.day, g.name, g.task, g.episodes[: args.limit]) for g in groups
        ]

    if args.list:
        total = sum(len(group.episodes) for group in groups)
        print(f"{len(groups)} dataset(s), {total} episodes\n")
        print(f"{'dataset':52s} {'code':5s} {'episodes':>9s} {'stations':>9s}")
        for group in groups:
            print(f"{group.name:52s} {group.code:5s} {len(group.episodes):9d} {len(group.stations):9d}")
        print(
            f"\nnot filed by task: {skipped['multi_code']} episodes in half-days listing "
            f"several task codes; {skipped['unscheduled']} outside the schedule; "
            f"{skipped['unknown_code']} with a code the catalogue lacks."
        )
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    # One driver per output tree. Two would hand the same dataset to two
    # processes and let them write the same directory; the child conversions
    # take no lock of their own, so this is where it has to be caught.
    if not args.only:
        lock_path = args.output / ".convert.lock"
        lock = lock_path.open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(
                f"Another conversion is already running on {args.output} "
                f"(lock: {lock_path}). Wait for it, or stop it first.",
                file=sys.stderr,
            )
            return 3
        lock.write(f"pid {os.getpid()} started {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        lock.flush()

    if args.jobs > 1 and not args.only:
        return 1 if run_in_parallel(groups, args) else 0

    for group in groups:
        result = convert_group(group, args.output, args.config, args.workers, args.overwrite)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
