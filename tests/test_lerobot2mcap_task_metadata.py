import json

import pandas as pd

from lerobot2mcap.dataset_info import DatasetInfo
from mcap2lerobot.converter import _write_task_metadata


def test_episode_manifest_carries_optional_task_identity_fields(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {}}), encoding="utf-8"
    )
    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"episode_index": 0, "length": 2, "tasks": ["Pick it up"]}]
    ).to_parquet(episodes_dir / "file-000.parquet", index=False)
    pd.DataFrame([{"task_index": 0, "task": "Pick it up"}]).to_parquet(
        meta / "tasks.parquet", index=False
    )
    (meta / "task_metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "episodes": [
                    {
                        "episode_index": 0,
                        "source_episode_id": "EP-1",
                        "task_id": "TASK-1",
                        "task_name": "pick",
                        "prompt": "Pick it up",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = DatasetInfo(meta / "info.json").episode_manifest(0)

    assert manifest["tasks"] == ["Pick it up"]
    assert manifest["task_index_to_task"] == {"0": "Pick it up"}
    assert {key: manifest[key] for key in ("source_episode_id", "task_id", "task_name", "prompt")} == {
        "source_episode_id": "EP-1",
        "task_id": "TASK-1",
        "task_name": "pick",
        "prompt": "Pick it up",
    }


def test_task_metadata_is_restored_from_roundtrip_episode_manifests(tmp_path):
    _write_task_metadata(
        tmp_path,
        [
            {
                "episode_index": 7,
                "source_episode_id": "EP-1",
                "task_id": "TASK-1",
                "task_name": "pick",
                "prompt": "Pick it up",
            }
        ],
    )

    payload = json.loads((tmp_path / "meta" / "task_metadata.json").read_text())
    assert payload == {
        "version": 1,
        "episodes": [
            {
                "episode_index": 0,
                "source_episode_id": "EP-1",
                "task_id": "TASK-1",
                "task_name": "pick",
                "prompt": "Pick it up",
            }
        ],
    }
