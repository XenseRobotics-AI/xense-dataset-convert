import os

from xense_dataset_convert._parallelism import available_cpu_count


def test_available_cpu_count_respects_process_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 5, 8})

    assert available_cpu_count() == 3


def test_available_cpu_count_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert available_cpu_count() == 6
