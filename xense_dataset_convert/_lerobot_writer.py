"""Shared helpers for configuring LeRobot dataset writers."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from ._parallelism import available_cpu_count

logger = logging.getLogger(__name__)


def writer_parallelism_kwargs(
    dataset_cls: Any,
    features: dict[str, dict[str, Any]],
    image_writer_threads: int,
    encoder_threads: int | None,
) -> dict[str, int]:
    """Configure LeRobot's asynchronous image writer and per-camera encoders."""
    parameters = inspect.signature(dataset_cls.create).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    def supports(name: str) -> bool:
        return name in parameters or accepts_kwargs

    video_count = sum(spec.get("dtype") == "video" for spec in features.values())
    kwargs: dict[str, int] = {}

    if supports("image_writer_threads"):
        kwargs["image_writer_threads"] = image_writer_threads
    elif image_writer_threads:
        logger.warning(
            "This lerobot does not expose image_writer_threads; the requested setting is ignored"
        )

    if video_count:
        cpu_count = available_cpu_count()
        resolved_encoder_threads = (
            encoder_threads if encoder_threads is not None else max(1, cpu_count // video_count)
        )
        if supports("encoder_threads"):
            kwargs["encoder_threads"] = resolved_encoder_threads
            logger.info(
                "Video encoding: %d camera(s) x %d thread(s), %d available CPU(s)",
                video_count,
                resolved_encoder_threads,
                cpu_count,
            )
        else:
            logger.warning(
                "This lerobot does not expose encoder_threads; video encoders use the codec default"
            )
    return kwargs
