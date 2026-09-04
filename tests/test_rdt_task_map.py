"""The RDT capture schedule as a per-episode task map.

The workbook is built here rather than read from disk, so the suite needs no
spreadsheet and no copy of the capture team's file.
"""

from __future__ import annotations

import datetime
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rdt_task_map import (  # noqa: E402
    EXCEL_EPOCH,
    episode_task,
    main,
    read_schedule,
)

TASK_ROWS = [
    ("A1", "临界压溃物品抓取，持续加压"),
    ("A2", "轻微扰动抗滑移"),
    ("A27", "物品左右交替，交到右手"),
]
SCHEDULE_ROWS = [
    (datetime.date(2026, 8, 11), "A2、A3、A27", "A1"),
    (datetime.date(2026, 8, 12), "", "A1"),
]


def _sheet(rows: list[list[str]]) -> str:
    cells = []
    for row_index, row in enumerate(rows, start=1):
        columns = "".join(
            f'<c r="{chr(65 + i)}{row_index}" t="inlineStr"><is><t>{value}</t></is></c>'
            for i, value in enumerate(row)
            if value != ""
        )
        cells.append(f"<row r=\"{row_index}\">{columns}</row>")
    return (
        '<?xml version="1.0"?><worksheet><sheetData>'
        + "".join(cells)
        + "</sheetData></worksheet>"
    )


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "schedule.xlsx"
    tasks = [["任务类型", "任务编号", "动作要求"]] + [
        ["常规任务", code, description] for code, description in TASK_ROWS
    ]
    schedule = [["日期", "上午", "下午"]] + [
        [str((day - EXCEL_EPOCH).days), morning, afternoon]
        for day, morning, afternoon in SCHEDULE_ROWS
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", _sheet(tasks))
        archive.writestr("xl/worksheets/sheet2.xml", _sheet(schedule))
    return path


def test_reads_codes_and_half_days(workbook: Path):
    tasks, schedule = read_schedule(workbook)
    assert tasks["A2"] == "轻微扰动抗滑移"
    assert schedule[datetime.date(2026, 8, 11)] == ("A2、A3、A27", "A1")


def test_morning_and_afternoon_pick_different_tasks(workbook: Path):
    tasks, schedule = read_schedule(workbook)
    morning = episode_task("ep_20260811_090603", tasks, schedule)
    afternoon = episode_task("ep_20260811_153703", tasks, schedule)
    # Several codes share a half-day: all are named, the first described.
    assert morning == "A2/A3/A27: 轻微扰动抗滑移"
    assert afternoon == "A1: 临界压溃物品抓取，持续加压"
    assert episode_task("ep_20260812_090000", tasks, schedule) is None  # empty morning
    assert episode_task("ep_20260901_090000", tasks, schedule) is None  # unscheduled day
    assert episode_task("not_an_episode", tasks, schedule) is None


def test_writes_a_map_for_a_rig_directory(workbook: Path, tmp_path: Path, capsys):
    rig = tmp_path / "035"
    for name in ("ep_20260811_090603", "ep_20260811_153703", "ep_20260901_120000"):
        (rig / name).mkdir(parents=True)
    output = tmp_path / "tasks.yaml"

    assert main(["--schedule", str(workbook), "--input", str(rig), "--output", str(output)]) == 0

    import yaml

    mapping = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert mapping == {
        "ep_20260811_090603": "A2/A3/A27: 轻微扰动抗滑移",
        "ep_20260811_153703": "A1: 临界压溃物品抓取，持续加压",
    }
    # The unscheduled recording is reported rather than given a made-up task.
    assert "ep_20260901_120000" in capsys.readouterr().err
