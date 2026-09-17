"""Stopping a conversion and finishing it later must cost nothing but time.

The contract is exact: a dataset built in two goes and one built in a single
run hold the same frames, the same statistics and the same metadata. Anything
weaker and resuming is a trap -- the seam would be invisible in the result and
show up later as a model trained on subtly different data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from hdf52lerobot.checkpoint import CHECKPOINT_NAME, Checkpoint
from hdf52lerobot.config import ConverterConfig
from hdf52lerobot.converter import Hdf5ToLeRobotConverter

from test_hdf52lerobot import FPS, _make_episode


EPISODES = 4


@pytest.fixture
def recordings(tmp_path: Path) -> list[Path]:
    """Four recordings, enough that a stop lands in the middle of the run."""
    root = tmp_path / "source"
    for index in range(EPISODES):
        _make_episode(root / f"ep_{index:04d}")
    return [root]


def config() -> ConverterConfig:
    return ConverterConfig(fps=FPS, gop=4, preset="ultrafast", task="resume test")


def convert_stopping_after(episodes: list[Path], output: Path, count: int) -> None:
    """Run a conversion that asks to stop once ``count`` episodes are done."""
    converter = Hdf5ToLeRobotConverter(episodes, output, config())
    original = converter._convert_episode
    state = {"done": 0}

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        state["done"] += 1
        if state["done"] >= count:
            # What the signal handler does; raising here instead would leave a
            # half-written episode, which is the case resuming must avoid.
            converter.request_stop()
        return result

    converter._convert_episode = counting
    with pytest.raises(KeyboardInterrupt):
        converter.convert()


def read_frames(root: Path) -> np.ndarray:
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    tables = [pq.read_table(path) for path in files]
    return np.concatenate(
        [np.stack(table.column("observation.state").to_numpy()) for table in tables]
    )


def test_resuming_reproduces_a_single_run(recordings, tmp_path):
    whole = Hdf5ToLeRobotConverter(recordings, tmp_path / "whole", config()).convert()

    split = tmp_path / "split"
    convert_stopping_after(recordings, split, 2)
    assert (split / CHECKPOINT_NAME).exists()
    assert not (split / "meta" / "info.json").exists(), "a partial dataset must not look finished"

    Hdf5ToLeRobotConverter(recordings, split, config()).convert()
    assert not (split / CHECKPOINT_NAME).exists(), "a finished dataset keeps no checkpoint"

    # The frames themselves, in order.
    assert np.array_equal(read_frames(whole), read_frames(split))

    # The statistics, which is where a running total would show a seam.
    assert json.loads((whole / "meta" / "stats.json").read_text()) == json.loads(
        (split / "meta" / "stats.json").read_text()
    )

    # And the metadata, bar the sizes, which depend on how the videos were cut.
    left = json.loads((whole / "meta" / "info.json").read_text())
    right = json.loads((split / "meta" / "info.json").read_text())
    for key in ("total_episodes", "total_frames", "total_tasks", "fps", "splits"):
        assert left[key] == right[key], key


def test_every_episode_is_converted_exactly_once(recordings, tmp_path):
    split = tmp_path / "split"
    convert_stopping_after(recordings, split, 1)
    Hdf5ToLeRobotConverter(recordings, split, config()).convert()

    sources = json.loads((split / "meta" / "sources.json").read_text())
    directories = [record["source_dir"] for record in sources["episodes"]]
    assert len(directories) == len(set(directories)) == EPISODES

    info = json.loads((split / "meta" / "info.json").read_text())
    assert info["total_episodes"] == EPISODES
    assert info["total_frames"] == len(read_frames(split))


def test_a_checkpoint_from_another_configuration_is_not_resumed(recordings, tmp_path):
    """Resuming under different settings would splice two datasets together."""
    split = tmp_path / "split"
    convert_stopping_after(recordings, split, 2)

    from dataclasses import replace

    other = replace(config(), fps=FPS * 2)
    Hdf5ToLeRobotConverter(recordings, split, other, overwrite=True).convert()

    info = json.loads((split / "meta" / "info.json").read_text())
    assert info["fps"] == FPS * 2
    assert info["total_episodes"] == EPISODES


def test_resume_can_be_refused(recordings, tmp_path):
    split = tmp_path / "split"
    convert_stopping_after(recordings, split, 2)
    with pytest.raises(FileExistsError):
        Hdf5ToLeRobotConverter(recordings, split, config(), resume=False).convert()


def test_a_stale_checkpoint_beside_a_finished_dataset_is_ignored(recordings, tmp_path):
    output = Hdf5ToLeRobotConverter(recordings, tmp_path / "done", config()).convert()
    shutil.copy(output / "meta" / "info.json", output / CHECKPOINT_NAME)
    assert Checkpoint.load(output) is None


def test_a_truncated_checkpoint_is_ignored(recordings, tmp_path):
    split = tmp_path / "split"
    convert_stopping_after(recordings, split, 2)
    (split / CHECKPOINT_NAME).write_text('{"version": 1, "converted": [')
    assert Checkpoint.load(split) is None


def test_the_first_signal_asks_to_stop_and_the_second_does_not():
    """One Ctrl-C means "at the next episode"; two mean now."""
    import os
    import signal

    from hdf52lerobot.checkpoint import InterruptRequest

    with InterruptRequest() as request:
        assert not request.requested
        os.kill(os.getpid(), signal.SIGINT)
        assert request.requested, "the process must survive the first signal"

    # The handler is put back on the way out, so nothing outside is affected.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_second_signal_is_left_to_python():
    import os
    import signal

    from hdf52lerobot.checkpoint import InterruptRequest

    with InterruptRequest() as request:
        os.kill(os.getpid(), signal.SIGINT)
        assert request.requested
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
