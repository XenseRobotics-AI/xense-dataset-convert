#!/usr/bin/env python3
"""Full parity run: pack every golden scenario with mcap2lerobot.xtac_pack and compare to Rust.

    python tools/xtac_pack_parity.py [--golden ~/taccap-cross/parity-golden] [--keep OUT_DIR]

JSON and mp4 are compared byte for byte, parquet logically (schema + bit-exact values). Exit 1 on
any difference. Scenarios without a ``lerobot/`` dataset (e.g. ``*-mcap``) must be refused.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcap2lerobot import xtac_pack as X


def compare(expected: Path, got: Path) -> list[str]:
    import numpy as np
    import pyarrow.parquet as pq

    problems = []
    want = {p.relative_to(expected) for p in expected.rglob("*") if p.is_file()}
    have = {p.relative_to(got) for p in got.rglob("*") if p.is_file()}
    problems += [f"missing {p}" for p in sorted(want - have)] + [f"extra {p}" for p in sorted(have - want)]
    for rel in sorted(want & have):
        a, b = expected / rel, got / rel
        if rel.suffix != ".parquet":
            if a.read_bytes() != b.read_bytes():
                problems.append(f"bytes differ {rel}")
            continue
        ta, tb = pq.read_table(a), pq.read_table(b)
        if not ta.schema.equals(tb.schema, check_metadata=False):
            problems.append(f"schema differs {rel}")
            continue
        for name in ta.column_names:
            x, y = ta.column(name).to_pylist(), tb.column(name).to_pylist()
            if ta.schema.field(name).type.equals(tb.schema.field(name).type) and "float" in str(ta.schema.field(name).type):
                same = np.array_equal(np.asarray(x, np.float32).view(np.uint32), np.asarray(y, np.float32).view(np.uint32))
            else:
                same = x == y
            if not same:
                problems.append(f"values differ {rel}:{name}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", type=Path, default=Path.home() / "taccap-cross" / "parity-golden")
    ap.add_argument("--keep", type=Path, help="keep the packed datasets here")
    args = ap.parse_args()
    work = args.keep or Path(tempfile.mkdtemp(prefix="xtac-parity-"))
    failed = 0
    for scenario in sorted(p for p in args.golden.iterdir() if (p / "train_mcap").is_dir()):
        inputs = sorted((scenario / "train_mcap").glob("*.train.mcap"))
        expected = scenario / "lerobot"
        if not any(p.is_file() for p in expected.rglob("*")):     # Rust refused it too
            try:
                X.pack(inputs, work / scenario.name)
            except X.TrainMcapError as error:
                print(f"OK   {scenario.name}: refused ({str(error)[:90]}…)")
                continue
            failed += 1
            print(f"FAIL {scenario.name}: has no Rust LeRobot output but was packed anyway")
            continue
        out = work / scenario.name
        shutil.rmtree(out, ignore_errors=True)
        result = X.pack(inputs, out)
        problems = compare(expected, out)
        failed += bool(problems)
        print(f"{'FAIL' if problems else 'OK  '} {scenario.name}: {result.episodes} episodes, {result.frames} frames"
              + "".join(f"\n       {p}" for p in problems))
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
