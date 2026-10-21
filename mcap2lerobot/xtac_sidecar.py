"""Deduplicated sidecar files: ``xumi_collection_devices.json`` v2 and ``taccap_extrinsics.json`` v4.

Port of taccap-core ``crates/taccap-lerobot/src/sidecar_v2.rs`` (43da99f). The training MCAP still
carries full per-episode v1 / v3 entries; compaction happens only when a file is written, and
``expand_*`` inverts it exactly. Ids are assigned in first-appearance order; equal content (as a
key-sorted ``serde_json::Value``) shares one id.
"""

from __future__ import annotations

import copy
import json
from typing import Any

COLLECTION_DEVICES_VERSION = 2
TACCAP_EXTRINSICS_VERSION = 4
EXPORT_PER_EPISODE = ("world_anchor", "initial_pose_check", "pose_interp_fallback_rows")
EXTRINSICS_PER_EPISODE = ("episode_index", "source_episode_id", "warnings")


class SidecarFormatError(ValueError):
    """A v2 / v4 file that does not expand: missing tables, dangling or non-string references."""


class _Table:
    def __init__(self, prefix: str) -> None:
        self.prefix, self.ids, self.entries = prefix, {}, {}

    def intern(self, value: Any) -> str:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if key not in self.ids:
            self.ids[key] = f"{self.prefix}_{len(self.entries)}"
            self.entries[self.ids[key]] = value
        return self.ids[key]


def _is_legacy(entry: dict) -> bool:
    return not (isinstance(entry.get("legacy"), bool) and entry["legacy"] is False)   # as_bool().unwrap_or(true)


def compact_collection_devices(entries: list[dict]) -> dict:
    devices, exports, bounds = _Table("devices"), _Table("export"), _Table("initial_pose_bounds")
    out = []
    for entry in copy.deepcopy(entries):
        entry = entry if isinstance(entry, dict) else {}
        if "devices" in entry:
            entry["devices"] = devices.intern(entry.pop("devices"))
        if "export" in entry:
            export = entry.pop("export")            # a non-object export is dropped, as in Rust
            if isinstance(export, dict):
                for key in EXPORT_PER_EPISODE:
                    if key in export:
                        value = export.pop(key)
                        if key == "initial_pose_check" and isinstance(value, dict) and "bounds" in value:
                            value["bounds"] = bounds.intern(value.pop("bounds"))
                        entry[key] = value
                entry["export"] = exports.intern(export)
        out.append(entry)
    legacy = sum(1 for e in entries if not isinstance(e, dict) or _is_legacy(e))
    return {"version": COLLECTION_DEVICES_VERSION, "total_episodes": len(entries),
            "snapshot_episodes": len(entries) - legacy, "legacy_episodes": legacy,
            "devices": devices.entries, "exports": exports.entries,
            "initial_pose_bounds": bounds.entries, "episodes": out}


def compact_taccap_extrinsics(entries: list[dict]) -> dict:
    profiles = _Table("extrinsics")
    out = []
    for entry in copy.deepcopy(entries):
        shared = entry if isinstance(entry, dict) else {}
        inline = {key: shared.pop(key) for key in EXTRINSICS_PER_EPISODE if key in shared}
        inline["profile"] = profiles.intern(shared)
        out.append(inline)
    return {"version": TACCAP_EXTRINSICS_VERSION, "profiles": profiles.entries, "episodes": out}


def expand_collection_devices(file: dict) -> list[dict]:
    """v2 (or v1) file → v1 entries."""
    episodes = file.get("episodes")
    if not isinstance(episodes, list):
        raise SidecarFormatError("collection devices file has no episodes")
    if file.get("version") != COLLECTION_DEVICES_VERSION:
        return copy.deepcopy(episodes)

    def lookup(table: str, ref: Any) -> Any:
        if not isinstance(ref, str):
            raise SidecarFormatError(f"{table} reference is not a string")
        if ref not in (file.get(table) or {}):
            raise SidecarFormatError(f"unknown {table} id {ref}")
        return copy.deepcopy(file[table][ref])

    out = []
    for entry in copy.deepcopy(episodes):
        if not isinstance(entry, dict):
            raise SidecarFormatError("episode entry is not an object")
        if "devices" in entry:
            entry["devices"] = lookup("devices", entry.pop("devices"))
        if "export" in entry:
            export = lookup("exports", entry.pop("export"))
            for key in EXPORT_PER_EPISODE:
                if key in entry:
                    value = entry.pop(key)
                    if key == "initial_pose_check" and isinstance(value, dict) and "bounds" in value:
                        value["bounds"] = lookup("initial_pose_bounds", value["bounds"])
                    export[key] = value
            entry["export"] = export
        out.append(entry)
    return out


def expand_taccap_extrinsics(file: dict) -> list[dict]:
    """v4 (or v3) file → v3 entries."""
    episodes = file.get("episodes")
    if not isinstance(episodes, list):
        raise SidecarFormatError("taccap extrinsics file has no episodes")
    if file.get("version") != TACCAP_EXTRINSICS_VERSION:
        return copy.deepcopy(episodes)
    out = []
    for entry in copy.deepcopy(episodes):
        if not isinstance(entry, dict) or not isinstance(entry.get("profile"), str):
            raise SidecarFormatError("extrinsics entry has no profile")
        ref = entry.pop("profile")
        if ref not in (file.get("profiles") or {}):
            raise SidecarFormatError(f"unknown extrinsics profile {ref}")
        out.append({**copy.deepcopy(file["profiles"][ref]), **entry})
    return out
