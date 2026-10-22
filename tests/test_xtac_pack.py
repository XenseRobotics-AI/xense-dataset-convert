"""xtac_train_mcap -> LeRobot: parity with taccap-core's Rust exporter.

The golden set (``XTAC_PARITY_GOLDEN``, default ``~/taccap-cross/parity-golden``) holds, per scenario,
the Rust-written training MCAPs and the LeRobot dataset the same build wrote. It is too large for the
repo; golden tests skip when it is absent. ``tools/xtac_pack_parity.py`` runs the same comparison
from the command line. The synthetic tests below need only ``mcap``.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from mcap2lerobot import xtac_pack as X

GOLDEN = Path(os.environ.get("XTAC_PARITY_GOLDEN", Path.home() / "taccap-cross" / "parity-golden"))
SCENARIOS = ["dual_claw_head-lerobot", "legacy_no_sidecar-lerobot"]
# JSON compared byte for byte; parquet logically (pyarrow and parquet-rs encode differently).
JSON_FILES = ["meta/info.json", "meta/stats.json", "meta/xumi_collection_devices.json",
              "meta/taccap_extrinsics.json"]
PARQUET_FILES = ["data/chunk-000/file-000.parquet", "meta/episodes/chunk-000/file-000.parquet",
                 "meta/tasks.parquet"]


def _golden(scenario: str) -> Path:
    root = GOLDEN / scenario
    if not (root / "train_mcap").is_dir():
        pytest.skip(f"parity golden {root} not present")
    return root


@pytest.fixture(scope="module", params=SCENARIOS)
def packed(request, tmp_path_factory):
    root = _golden(request.param)
    out = tmp_path_factory.mktemp(request.param) / "dataset"
    X.pack(sorted((root / "train_mcap").glob("*.train.mcap")), out)
    return root / "lerobot", out


def test_json_is_byte_identical(packed):
    expected, got = packed
    for rel in JSON_FILES:
        assert (got / rel).read_bytes() == (expected / rel).read_bytes(), rel


def test_parquet_is_logically_identical(packed):
    import pyarrow.parquet as pq

    expected, got = packed
    for rel in PARQUET_FILES:
        want, have = pq.read_table(expected / rel), pq.read_table(got / rel)
        assert have.schema.equals(want.schema, check_metadata=False), rel
        assert (have.schema.metadata or {}).get(b"pandas") == (want.schema.metadata or {}).get(b"pandas"), rel
        for name in want.column_names:
            a, b = want.column(name).to_pylist(), have.column(name).to_pylist()
            if name in ("observation.state", "action", "observation.tracker_confidence", "timestamp"):
                a = np.asarray(a, dtype=np.float32).view(np.uint32)       # bit-exact, not ≈
                b = np.asarray(b, dtype=np.float32).view(np.uint32)
                assert np.array_equal(a, b), f"{rel}:{name}"
            else:
                assert a == b, f"{rel}:{name}"


def test_mp4_is_byte_identical(packed):
    expected, got = packed
    files = sorted(p.relative_to(expected) for p in expected.rglob("*.mp4"))
    assert files == sorted(p.relative_to(got) for p in got.rglob("*.mp4"))
    for rel in files:
        assert (got / rel).read_bytes() == (expected / rel).read_bytes(), rel


def test_same_files_as_rust(packed):
    expected, got = packed
    assert ({p.relative_to(expected) for p in expected.rglob("*") if p.is_file()}
            == {p.relative_to(got) for p in got.rglob("*") if p.is_file()})


def test_reference_camera_timestamps_is_refused():
    root = _golden("dual_claw_head-mcap")
    with pytest.raises(X.TrainMcapError, match="reference_camera_timestamps"):
        X.read_train_mcap(next((root / "train_mcap").glob("*.train.mcap")))


# ── synthetic: no golden needed ────────────────────────────────────────────────


def _write_train_mcap(path: Path, *, timing="fixed_rate_grid", started_ms=1792602557684, episode_id="EP-1",
                      task_name="t", rows=3, omit=(), tcp=None):
    from mcap.writer import Writer

    names = [f"d{i}" for i in range(2)]
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema = writer.register_schema("x", "jsonschema", b"{}")
        channels = {topic: writer.register_channel(topic, "json", schema)
                    for topic in ("/train/state", "/train/action", "/train/tracker_confidence")
                    if topic not in omit}
        for i in range(rows):
            for channel in channels.values():
                writer.add_message(channel, log_time=i, publish_time=i, sequence=i + 1,
                                   data=json.dumps({"grid_index": i, "values": [float(i), 0.5]}).encode())
        train = {"timing": timing, "fps": "30", "rows": str(rows), "state_dim": "2",
                 "dimension_names": ",".join(names), "robot_type": "xtac_umi_g1"}
        if tcp is not None:
            train["tcp_frame_convention"] = tcp
        writer.add_metadata("xense_train_mcap", train)
        writer.add_metadata("episode", {"episode_id": episode_id, "started_at_ms": str(started_ms),
                                        "capture_mode": "dual_claw_head", "task_name": task_name, "prompt": "p"})
        if "stats" not in omit:
            writer.add_metadata("xense_lerobot_video_stats", {"json": '{"version":1,"topics":{}}'})
        writer.add_metadata("xumi_lerobot_episode", {"xumi_collection_devices": '{"episode_index":7,"legacy":false}',
                                                     "taccap_extrinsics": '{"episode_index":7}'})
        writer.finish()
    return path


def test_synthetic_reference_timing_is_refused(tmp_path):
    with pytest.raises(X.TrainMcapError, match="reference_camera_timestamps"):
        X.read_train_mcap(_write_train_mcap(tmp_path / "a.train.mcap", timing="reference_camera_timestamps"))


@pytest.mark.parametrize("omit", ["/train/tracker_confidence", "stats"])
def test_pre_2e43d63_training_mcaps_are_refused(tmp_path, omit):
    with pytest.raises(X.TrainMcapError, match="2e43d63"):
        X.read_train_mcap(_write_train_mcap(tmp_path / "a.train.mcap", omit=(omit,)))


def test_episodes_are_ordered_by_start_time_and_reindexed(tmp_path):
    late = X.read_train_mcap(_write_train_mcap(tmp_path / "b.train.mcap", started_ms=1792602600000, episode_id="EP-B"))
    early = X.read_train_mcap(_write_train_mcap(tmp_path / "a.train.mcap", started_ms=1792602500000, episode_id="EP-A"))
    assert [e.episode_id for e in X.order_episodes([late, early])] == ["EP-A", "EP-B"]
    out = tmp_path / "out"
    X.pack([tmp_path / "b.train.mcap", tmp_path / "a.train.mcap"], out)
    devices = json.loads((out / "meta/xumi_collection_devices.json").read_text())
    assert [e["episode_index"] for e in devices["episodes"]] == [0, 1]       # rewritten from 7


def test_inputs_from_two_days_are_refused(tmp_path):
    a = X.read_train_mcap(_write_train_mcap(tmp_path / "a.train.mcap", started_ms=1792602500000))
    b = X.read_train_mcap(_write_train_mcap(tmp_path / "b.train.mcap", started_ms=1792602500000 + 86_400_000))
    with pytest.raises(X.TrainMcapError, match="datasets"):
        X.order_episodes([a, b])


@pytest.mark.parametrize("value,text", [
    (3136.0, "3136.0"), (1e16, "10000000000000000.0"), (1.2198561307741329e-05, "0.000012198561307741329"),
    (5.477736067405203e-6, "5.477736067405203e-6"), (1e22, "1e22"), (0.0, "0.0"), (-0.25, "-0.25"),
])
def test_floats_format_like_serde_json(value, text):
    assert X.format_f64(value) == text


def test_json_has_no_trailing_newline_and_sorted_value_keys(tmp_path):
    X._write_json(tmp_path / "a.json", X.sort_keys({"b": 1, "a": {"z": [], "y": {}}}))
    assert (tmp_path / "a.json").read_text() == '{\n  "a": {\n    "y": {},\n    "z": []\n  },\n  "b": 1\n}'


def test_annex_b_split_strips_trailing_zeros():
    au = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x01\x65\xaa\x00\x00"
    assert X.annex_b_nal_units(au) == [b"\x09\xf0", b"\x65\xaa"]


def test_file_rolling_flushes_before_reaching_the_threshold():
    assert X._roll([60, 30, 20, 50], 100) == [(0, 0), (0, 0), (0, 1), (0, 1)]


def test_stats_sum_sequentially_not_pairwise():
    values = np.array([1e16, 1.0, -1e16, 1.0] * 3, dtype=np.float64)
    sequential = 0.0
    for v in values:
        sequential += v
    assert X._numeric_stats(values)["mean"] == [sequential / len(values)]


def test_mux_layout_is_ftyp_mdat_moov(tmp_path):
    golden = GOLDEN / "dual_claw_head-lerobot" / "lerobot" / "videos" / "observation.images.left_wrist"
    mp4 = next(golden.rglob("*.mp4"), None) if golden.is_dir() else None
    if mp4 is None:
        pytest.skip("parity golden not present")
    data = mp4.read_bytes()
    kinds, offset = [], 0
    while offset < len(data):
        size, kind = struct.unpack(">I4s", data[offset:offset + 8])
        kinds.append(kind)
        offset += size
    assert kinds == [b"ftyp", b"mdat", b"moov"]


@pytest.mark.parametrize("tcp", [None, "mirrored-v1"])
def test_tcp_frame_convention_is_passed_through_after_collection_stack(tmp_path, tcp):
    X.pack([_write_train_mcap(tmp_path / "a.train.mcap", tcp=tcp)], tmp_path / "out")
    keys = list(json.loads((tmp_path / "out/meta/info.json").read_text()))
    if tcp is None:
        assert "tcp_frame_convention" not in keys                    # pre-556dd3d MCAPs: no claim made
    else:
        assert keys[keys.index("collection_stack") + 1] == "tcp_frame_convention"
        assert json.loads((tmp_path / "out/meta/info.json").read_text())["tcp_frame_convention"] == tcp


def test_inputs_with_different_tcp_conventions_are_refused(tmp_path):
    a = X.read_train_mcap(_write_train_mcap(tmp_path / "a.train.mcap", episode_id="EP-A", tcp="mirrored-v1"))
    b = X.read_train_mcap(_write_train_mcap(tmp_path / "b.train.mcap", episode_id="EP-B", started_ms=1792602600000))
    with pytest.raises(X.TrainMcapError, match="tcp_frame_convention"):
        X.order_episodes([a, b])


def test_tactile_recipes_from_older_mcaps_are_dropped_but_wrist_passes_through():
    e = X.TrainEpisode.__new__(X.TrainEpisode)      # only the two fields _rectify reads
    e.calibration = {
        "/camera/left/tactile_1/h264": {"rectified": True, "rectify_kind": "tactile", "rectify_config": {"g": 1}},
        "/camera/left/fisheye/h264": {"rectified": True, "rectify_kind": "fisheye", "rectify_config": {"k": 2}},
    }
    e.channel_metadata = {}
    assert X._rectify([e], "observation.images.left_tactile_left") is None
    assert X._rectify([e], "observation.images.left_wrist") == ("fisheye", {"k": 2})
