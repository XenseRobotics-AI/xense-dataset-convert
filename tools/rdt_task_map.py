#!/usr/bin/env python3
"""Turn the RDT capture schedule into a per-episode task map for hdf52lerobot.

The RDT recordings carry no task text. What identifies a session's task is
*when* it was recorded: the capture team's `采集任务时间安排表.xlsx` gives a
task code per half-day (morning / afternoon), and its first sheet spells each
code out. A recording directory is named `ep_YYYYMMDD_HHMMSS`, so its date and
whether it started before noon are enough to look the task up.

    python tools/rdt_task_map.py \
        --schedule ~/Documents/TacVerse-RDT-Rules/采集任务时间安排表.xlsx \
        --input /archive/TacVerse-RDT-Raw/035 --input /archive/TacVerse-RDT-Raw/036 \
        --output configs/tacverse_rdt_tasks.yaml

    hdf52lerobot --config configs/tacverse_rdt.yaml \
        --tasks configs/tacverse_rdt_tasks.yaml ...

The workbook is read with the standard library alone (it is a zip of XML), so
this needs no spreadsheet dependency in an environment that only converts data.
A half-day that lists several codes yields all of them, because the session
really did cover several: `A2/A3/A27: <description of A2>` names what happened
without inventing a single label for it.
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
import zipfile
from pathlib import Path

# Excel's 1900 date system counts from this day (its leap-year bug included).
EXCEL_EPOCH = datetime.date(1899, 12, 30)
EPISODE_NAME = re.compile(r"^ep_(\d{8})_(\d{6})$")
CODE_SEPARATORS = re.compile(r"[、,，/]")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    document = archive.read("xl/sharedStrings.xml").decode("utf-8")
    return [
        html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", item, re.S)))
        for item in re.findall(r"<si>(.*?)</si>", document, re.S)
    ]


def _rows(archive: zipfile.ZipFile, sheet: str, shared: list[str]) -> list[dict[str, str]]:
    """Cells of one worksheet as ``{column letter: text}`` per row."""
    document = archive.read(sheet).decode("utf-8")
    rows: list[dict[str, str]] = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", document, re.S):
        cells: dict[str, str] = {}
        for attributes, inner in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            reference = re.search(r'r="([A-Z]+)\d+"', attributes)
            if not reference:
                continue
            kind = re.search(r't="(\w+)"', attributes)
            # A cell holds either a value (a number, or an index into the shared
            # string table) or, when the writer chose not to share strings, the
            # text inline.
            value = re.search(r"<v>(.*?)</v>", inner or "", re.S)
            if value:
                text = value.group(1)
                if kind and kind.group(1) == "s":
                    text = shared[int(text)]
            else:
                inline = re.search(r"<is>.*?<t[^>]*>(.*?)</t>", inner or "", re.S)
                if not inline:
                    continue
                text = inline.group(1)
            cells[reference.group(1)] = html.unescape(text).strip()
        rows.append(cells)
    return rows


def read_schedule(path: Path) -> tuple[dict[str, str], dict[datetime.date, tuple[str, str]]]:
    """``({code: description}, {date: (morning codes, afternoon codes)})``."""
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheets = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if len(sheets) < 2:
            raise ValueError(f"{path}: expected a task sheet and a schedule sheet")
        tasks = {}
        for row in _rows(archive, sheets[0], shared)[1:]:
            code, description = row.get("B", ""), row.get("C", "")
            if code and description:
                tasks[code] = " ".join(description.split())
        schedule = {}
        for row in _rows(archive, sheets[1], shared)[1:]:
            serial = row.get("A", "")
            if not serial.isdigit():
                continue
            day = EXCEL_EPOCH + datetime.timedelta(days=int(serial))
            schedule[day] = (row.get("B", ""), row.get("C", ""))
    return tasks, schedule


def episode_task(
    name: str,
    tasks: dict[str, str],
    schedule: dict[datetime.date, tuple[str, str]],
) -> str | None:
    """Task text for one ``ep_YYYYMMDD_HHMMSS`` directory, or None if unscheduled."""
    match = EPISODE_NAME.match(name)
    if not match:
        return None
    stamp = datetime.datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
    half_days = schedule.get(stamp.date())
    if not half_days:
        return None
    codes = [
        code.strip()
        for code in CODE_SEPARATORS.split(half_days[0 if stamp.hour < 12 else 1])
        if code.strip()
    ]
    if not codes:
        return None
    described = [code for code in codes if code in tasks]
    if not described:
        return None
    # Several codes share a half-day; name them all, then describe the first.
    return f"{'/'.join(codes)}: {tasks[described[0]]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", required=True, help="采集任务时间安排表.xlsx")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="A rig directory of ep_* recordings, or one recording. Repeatable.",
    )
    parser.add_argument("--output", help="Write the map here (default: stdout).")
    args = parser.parse_args(argv)

    tasks, schedule = read_schedule(Path(args.schedule).expanduser())
    episodes: list[Path] = []
    for entry in args.input:
        path = Path(entry).expanduser().resolve()
        episodes.extend(
            [path] if EPISODE_NAME.match(path.name) else sorted(p for p in path.iterdir() if p.is_dir())
        )

    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for episode in episodes:
        task = episode_task(episode.name, tasks, schedule)
        if task is None:
            unmatched.append(episode.name)
        else:
            mapping[episode.name] = task

    lines = [
        "# Per-episode task text for the TacVerse-RDT recordings.",
        "#",
        "# Generated by tools/rdt_task_map.py from the capture team's schedule:",
        "# a recording's date and half-day give the task code(s), which the",
        "# workbook's first sheet spells out. Regenerate rather than edit.",
        "",
    ]
    for name, task in sorted(mapping.items()):
        lines.append(f"{name}: {json.dumps(task, ensure_ascii=False)}")
    document = "\n".join(lines) + "\n"

    if args.output:
        Path(args.output).expanduser().write_text(document, encoding="utf-8")
        print(f"{len(mapping)} episode(s) -> {args.output}", file=sys.stderr)
    else:
        print(document, end="")
    if unmatched:
        print(f"no schedule entry for {len(unmatched)}: {', '.join(unmatched)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
