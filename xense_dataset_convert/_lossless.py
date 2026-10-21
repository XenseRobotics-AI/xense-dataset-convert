"""Checksummed source preservation for conversions between unequal schemas.

XTac's sensor streams cannot all be represented by LeRobot's frame table.
Keep the original bytes as well as the usable projection; never silently replay
a stale source after somebody edits that projection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from time import monotonic
from types import SimpleNamespace
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

from ._progress import Progress

logger = logging.getLogger(__name__)

ATTACHMENT_PREFIX = "xense/lerobot/"
MANIFEST_ATTACHMENT = ATTACHMENT_PREFIX + "manifest.json"
SOURCE_DIRECTORY = "meta/source_mcap"
BLOCK_SIZE = 8 * 1024 * 1024


def update_message_hash(hasher: Any, schema: Any, channel: Any, message: Any) -> None:
    header = [schema.name if schema else "", schema.encoding if schema else "",
              hashlib.sha256(schema.data if schema else b"").hexdigest(),
              channel.topic, channel.message_encoding, channel.metadata,
              message.log_time, message.publish_time, message.sequence]
    for data in (json.dumps(header, sort_keys=True).encode(), message.data):
        hasher.update(len(data).to_bytes(8, "little"))
        hasher.update(data)


def projection_digest(path: Path) -> str:
    from mcap.reader import make_reader

    with path.open("rb") as stream:
        progress = Progress(logger, f"Verifying XTac messages: {path.name}", unit="messages")
        reader = make_reader(stream)
        hasher = hashlib.sha256()
        metadata = [(item.name, item.metadata) for item in reader.iter_metadata()]
        count = 0
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            update_message_hash(hasher, schema, channel, message)
            count += 1
            progress.update(count)
        progress.update(count, done=True)
        return _combine_hashes(metadata, hasher.hexdigest())


def tracked_writer(output: Any) -> Any:
    """Track the view separately from attachments, to reject edits to either."""
    from mcap.writer import Writer

    class TrackedWriter(Writer):
        def __init__(self):
            super().__init__(output)
            self.schemas, self.channels, self.metadata_records = {}, {}, []
            self.messages_hash = hashlib.sha256()

        def register_schema(self, name, encoding, data):
            identifier = super().register_schema(name, encoding, data)
            self.schemas[identifier] = SimpleNamespace(name=name, encoding=encoding, data=data)
            return identifier

        def register_channel(self, topic, message_encoding, schema_id, metadata=None):
            identifier = super().register_channel(topic, message_encoding, schema_id, metadata or {})
            self.channels[identifier] = SimpleNamespace(topic=topic, message_encoding=message_encoding,
                                                        schema_id=schema_id, metadata=metadata or {})
            return identifier

        def add_metadata(self, name, data):
            self.metadata_records.append((name, data))
            return super().add_metadata(name, data)

        def add_message(self, channel_id, log_time, data, publish_time, sequence=0):
            channel = self.channels[channel_id]
            update_message_hash(self.messages_hash, self.schemas[channel.schema_id], channel,
                                SimpleNamespace(log_time=log_time, publish_time=publish_time,
                                                sequence=sequence, data=data))
            return super().add_message(channel_id, log_time, data, publish_time, sequence)

        def projection_digest(self):
            # A copy lets metadata form a separate prefix, regardless of where
            # it was physically written relative to messages.
            return _combine_hashes(self.metadata_records, self.messages_hash.hexdigest())

    return TrackedWriter()


def _combine_hashes(metadata: list, messages: str) -> str:
    return hashlib.sha256(json.dumps([metadata, messages], sort_keys=True).encode()).hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(name: str) -> Path:
    """Accept portable relative paths, including no traversal or Windows drives."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe source path: {name!r}")
    if str(path) != name or name == ".":
        raise ValueError(f"Non-canonical source path: {name!r}")
    return Path(*path.parts)


def inventory(root: Path, exclude: str | None = None) -> dict[str, Any]:
    """Hash files without loading videos into memory; record empty directories too."""
    files, directories = [], []
    progress = Progress(logger, f"SHA-256 file inventory: {root}", unit="files")
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if exclude and (name == exclude or name.startswith(exclude + "/")):
            continue
        safe_path(name)
        if path.is_symlink():
            raise ValueError(f"Lossless conversion requires regular files, found symlink: {path}")
        if path.is_dir():
            directories.append(name)
        elif path.is_file():
            logger.debug("Hashing %s", path)
            files.append({"path": name, "size": path.stat().st_size, "sha256": digest(path)})
            progress.update(len(files))
        else:
            raise ValueError(f"Unsupported source file: {path}")
    progress.update(len(files), done=True)
    return {"files": files, "directories": directories}


def build_manifest(root: Path, episodes: Sequence[int]) -> dict[str, Any]:
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Lossless export needs a nonempty, unique episode selection")
    manifest = {"version": 1, "kind": "lerobot", "episodes": list(episodes), **inventory(root)}
    part_number = 0
    for entry in manifest["files"]:
        parts = []
        for offset in range(0, entry["size"], BLOCK_SIZE):
            parts.append({
                "name": f"{ATTACHMENT_PREFIX}parts/{part_number:09d}",
                "episode": episodes[part_number % len(episodes)],
                "offset": offset,
                "size": min(BLOCK_SIZE, entry["size"] - offset),
            })
            part_number += 1
        entry["parts"] = parts
    return manifest


def add_source_attachments(writer: Any, root: Path, manifest: dict, episode: int) -> None:
    def add(name: str, data: bytes, media_type: str) -> None:
        writer.add_attachment(0, 0, name, media_type, data)

    add(MANIFEST_ATTACHMENT, json.dumps({"manifest": manifest, "episode": episode,
                                        "projection_sha256": writer.projection_digest()}).encode(),
        "application/json")
    total = sum(1 for entry in manifest["files"] for part in entry["parts"] if part["episode"] == episode)
    progress = Progress(logger, f"Episode {episode}: preserving source attachments", total, "blocks")
    written = 0
    for entry in manifest["files"]:
        with (root / safe_path(entry["path"])).open("rb") as stream:
            for part in entry["parts"]:
                if part["episode"] != episode:
                    continue
                stream.seek(part["offset"])
                data = stream.read(part["size"])
                if len(data) != part["size"]:
                    raise ValueError(f"Source changed during conversion: {entry['path']}")
                add(part["name"], data, "application/octet-stream")
                written += 1
                progress.update(written)
    progress.update(written, done=True)


def attachments(path: Path) -> Iterator[Any]:
    from mcap.reader import make_reader

    with path.open("rb") as stream:
        yield from make_reader(stream).iter_attachments()


def source_manifest(path: Path) -> dict | None:
    logger.info("Inspecting source manifest: %s", path.name)
    for attachment in attachments(path):
        if attachment.name == MANIFEST_ATTACHMENT:
            return json.loads(attachment.data)
    return None


def restore_lerobot(inputs: Sequence[Path], output: Path) -> bool:
    """Restore a complete, verified export atomically, or return False for raw MCAP."""
    started = monotonic()
    logger.info("Checking preserved LeRobot source in %d MCAP files", len(inputs))
    records = [source_manifest(path) for path in inputs]
    if not any(record is not None for record in records):
        return False
    if not all(record is not None for record in records):
        raise ValueError("Cannot mix lossless LeRobot exports with other MCAP files")
    manifest = records[0]["manifest"]
    if manifest.get("version") != 1 or manifest.get("kind") != "lerobot":
        raise ValueError("Unsupported lossless manifest version or kind")
    if any(record["manifest"] != manifest for record in records):
        raise ValueError("MCAP files belong to different source snapshots")
    if sorted(record["episode"] for record in records) != sorted(manifest["episodes"]):
        raise ValueError("Incomplete lossless export: provide all exported episodes, or disable source preservation")
    logger.info("Restoring preserved LeRobot files: %d episodes, %d files, %.2f MiB; output=%s",
                len(inputs), len(manifest["files"]), sum(entry["size"] for entry in manifest["files"]) / 1024**2, output)
    for index, (path, record) in enumerate(zip(inputs, records), 1):
        logger.info("Episode %d/%d: checking XTac projection integrity", index, len(inputs))
        if projection_digest(path) != record["projection_sha256"]:
            raise ValueError(f"XTac projection checksum mismatch: {path}")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xense-restore-", dir=output.parent) as staging:
        root = Path(staging) / "dataset"
        root.mkdir()
        destinations: dict[str, tuple[Path, dict]] = {}
        file_names: set[str] = set()
        for entry in manifest["files"]:
            relative = safe_path(entry["path"])
            if entry["path"] in file_names:
                raise ValueError("Duplicate source file in manifest")
            file_names.add(entry["path"])
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            offset = 0
            for part in entry["parts"]:
                if part["name"] in destinations or part["offset"] != offset or not 0 < part["size"] <= BLOCK_SIZE:
                    raise ValueError("Invalid or duplicate attachment part")
                destinations[part["name"]] = (path, part)
                offset += part["size"]
            if offset != entry["size"]:
                raise ValueError("Invalid source file length in manifest")
        for directory in manifest["directories"]:
            (root / safe_path(directory)).mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        progress = Progress(logger, "Restoring source attachments", len(destinations), "blocks")
        for index, (source, record) in enumerate(zip(inputs, records), 1):
            logger.info("Episode %d/%d: extracting source attachments from %s", index, len(inputs), source.name)
            for attachment in attachments(source):
                if attachment.name not in destinations:
                    continue
                path, part = destinations[attachment.name]
                if attachment.name in seen or part["episode"] != record["episode"] or len(attachment.data) != part["size"]:
                    raise ValueError("Duplicate, misplaced or truncated source attachment")
                seen.add(attachment.name)
                with path.open("r+b") as stream:
                    stream.seek(part["offset"])
                    stream.write(attachment.data)
                progress.update(len(seen))
        if seen != set(destinations):
            raise ValueError("Missing source attachments in lossless export")
        progress.update(len(seen), done=True)
        progress = Progress(logger, "Verifying restored source SHA-256", len(manifest["files"]), "files")
        for index, entry in enumerate(manifest["files"], 1):
            if digest(root / safe_path(entry["path"])) != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {entry['path']}")
            progress.update(index)
        progress.update(len(manifest["files"]), done=True)
        logger.info("All checks passed; publishing restored LeRobot dataset to %s", output)
        root.rename(output)
    logger.info("LeRobot source restoration completed: %d episodes, %d files, elapsed %.1fs, output=%s",
                len(inputs), len(manifest["files"]), monotonic() - started, output)
    return True


def preserve_mcap(inputs: Sequence[Path], root: Path) -> None:
    """Keep every original channel, record, timestamp and compressed video byte."""
    target = root / SOURCE_DIRECTORY
    target.mkdir(parents=True, exist_ok=False)
    files = []
    common = Path(os.path.commonpath([source.resolve().parent for source in inputs]))
    progress = Progress(logger, "Preserving original MCAP files", len(inputs), "files")
    for index, source in enumerate(inputs):
        logger.info("Preserving MCAP %d/%d: %s (%.2f MiB)", index + 1, len(inputs), source.name,
                    source.stat().st_size / 1024**2)
        name = f"{index:06d}/{source.name}"
        destination = target / safe_path(name)
        destination.parent.mkdir()
        shutil.copyfile(source, destination)
        files.append({"path": name, "name": source.resolve().relative_to(common).as_posix(), "sha256": digest(destination)})
        progress.update(index + 1)
    progress.update(len(inputs), done=True)
    record = {"version": 1, "files": files, "projection": inventory(root, SOURCE_DIRECTORY)}
    (target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    logger.info("Original MCAP files and projection checksums saved to %s", target)


def restore_mcap(root: Path, output: Path) -> bool:
    started = monotonic()
    target = root / SOURCE_DIRECTORY
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        return False
    record = json.loads(manifest_path.read_text())
    if record.get("version") != 1:
        raise ValueError("Unsupported preserved MCAP manifest version")
    logger.info("Restoring %d original MCAP files; checking LeRobot projection first", len(record["files"]))
    if inventory(root, SOURCE_DIRECTORY) != record["projection"]:
        raise ValueError("LeRobot projection changed; disable source preservation to export edited data")
    names = [entry["name"] for entry in record["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate original MCAP paths")
    for index, entry in enumerate(record["files"], 1):
        logger.info("Verifying original MCAP SHA-256 %d/%d: %s", index, len(record["files"]), entry["name"])
        safe_path(entry["name"])
        source = target / safe_path(entry["path"])
        if source.is_symlink() or digest(source) != entry["sha256"]:
            raise ValueError(f"Preserved MCAP checksum mismatch: {source}")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xense-restore-", dir=output.parent) as staging:
        destination = Path(staging) / "mcap"
        destination.mkdir()
        for index, entry in enumerate(record["files"], 1):
            logger.info("Restoring original MCAP %d/%d: %s", index, len(record["files"]), entry["name"])
            path = destination / entry["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target / entry["path"], path)
        destination.rename(output)
    logger.info("Original MCAP restoration completed: %d files, elapsed %.1fs, output=%s",
                len(record["files"]), monotonic() - started, output)
    return True
