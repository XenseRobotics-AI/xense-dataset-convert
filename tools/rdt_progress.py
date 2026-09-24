#!/usr/bin/env python3
"""Show how far the TacVerse-RDT bulk conversion has got.

A conversion writes nothing a reader can count until it finalises -- the
parquet and the videos are still being appended -- so progress for a dataset
still running comes from its log: the converter writes one `grid 20Hz:` line
per episode it finishes. A dataset that *has* finalised is read from its own
metadata instead, which is both authoritative and immune to the logs being
appended across runs.

An episode that cannot be converted is not outstanding work, so the percentage
counts converted *and* skipped against the total, and reaches 100% when the run
is over. Rate and finish time are shown only while something is still running:
after that, elapsed time keeps growing while nothing is produced, and a rate
computed from it says less every minute.

    python tools/rdt_progress.py                     # one look
    watch -n 60 python tools/rdt_progress.py         # keep it on screen
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rdt_convert import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_RAW,
    DEFAULT_SCHEDULE,
    REPO,
    build_groups,
    load_catalogue,
)
from tools.rdt_task_map import read_schedule  # noqa: E402

EPISODE_DONE = re.compile(r"grid \d+Hz:")
SKIPPED = re.compile(r"\[warning\] .*: skipped \(")


def directory_bytes(path: Path) -> int:
    try:
        out = subprocess.run(
            ["du", "-sb", str(path)], capture_output=True, text=True, timeout=120
        )
        return int(out.stdout.split()[0]) if out.returncode == 0 else 0
    except Exception:
        return 0


def human(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "--"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--catalogue", type=Path, default=REPO / "configs/tacverse_rdt_catalogue.yaml")
    parser.add_argument("--prefix", default="xtac-rdt-umi")
    parser.add_argument("--no-size", action="store_true", help="Skip the disk figures (they walk the tree).")
    args = parser.parse_args(argv)

    _, schedule = read_schedule(args.schedule)
    groups, _ = build_groups(args.raw, schedule, load_catalogue(args.catalogue), args.prefix)
    expected = {group.name: len(group.episodes) for group in groups}

    log_dir = args.output / ".convert-logs"
    # When the run began. The lock file records it; the logs cannot, since
    # their inode times move every time a line is appended.
    started_at = None
    lock = args.output / ".convert.lock"
    if lock.exists():
        stamp = re.search(r"started (\S+)", lock.read_text(encoding="utf-8", errors="replace"))
        if stamp:
            started_at = datetime.datetime.fromisoformat(stamp.group(1)).timestamp()
    rows = []
    done_total = expected_total = skipped_total = 0

    for name, total in sorted(expected.items(), key=lambda kv: -kv[1]):
        log = log_dir / f"{name}.log"
        info = args.output / name / "meta" / "info.json"
        sources = args.output / name / "meta" / "sources.json"
        done = skipped = 0
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            done = len(EPISODE_DONE.findall(text))
            skipped = len(SKIPPED.findall(text))
        if info.exists():
            state = "done"
            # The dataset's own record, not the log: logs are appended across
            # runs, so an interrupted first attempt would be counted twice.
            done = json.loads(info.read_text(encoding="utf-8"))["total_episodes"]
            if sources.exists():
                skipped = len(
                    json.loads(sources.read_text(encoding="utf-8")).get("skipped", [])
                )
        elif log.exists() and done:
            state = "running"
        elif log.exists():
            state = "starting"
        else:
            state = "queued"
        rows.append((name, state, done, skipped, total))
        done_total += done
        skipped_total += skipped
        expected_total += total

    finished = all(state == "done" for _, state, _, _, _ in rows)
    elapsed = time.time() - started_at if started_at else 0
    rate = done_total / elapsed if elapsed > 0 else 0
    # What is left to do, not what was skipped: a skipped episode is settled.
    outstanding = expected_total - done_total - skipped_total
    remaining = outstanding / rate if rate > 0 else float("nan")

    print(f"{'dataset':46s} {'state':9s} {'episodes':>16s} {'skipped':>8s}  progress")
    for name, state, done, skipped, total in rows:
        share = (done + skipped) / total if total else 0
        bar = "#" * round(share * 24)
        print(f"{name:46s} {state:9s} {done:7d} / {total:6d} {skipped:8d}  {bar:<24s} {share*100:5.1f}%")

    settled = done_total + skipped_total
    print(
        f"\n{settled} / {expected_total} episodes settled "
        f"({100 * settled / expected_total:.1f}%): {done_total} converted, "
        f"{skipped_total} skipped as unusable"
    )
    if finished:
        print(f"finished: all {len(rows)} dataset(s) converted in {human(elapsed)}")
    elif rate > 0:
        finish = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        print(
            f"running {human(elapsed)}, {rate * 3600:.0f} episodes/hour, "
            f"about {human(remaining)} left (≈ {finish.strftime('%m-%d %H:%M')})"
        )
    if not args.no_size:
        written = directory_bytes(args.output)
        if written:
            projected = written / done_total * expected_total if done_total else 0
            print(f"written {written / 2**30:.1f} GiB, projected total {projected / 2**30:.0f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
