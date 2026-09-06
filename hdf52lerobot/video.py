"""Transcode source videos into the aggregated, constant-frame-rate mp4 files
that a LeRobot v3.0 dataset expects.

Sources are re-encoded rather than remuxed: the rig's cameras write H.264 with
GOPs of 200+ frames, which makes LeRobot's random-access reads crawl, and the
tactile sensors write FFV1, which mp4 cannot carry at all. Re-encoding also
makes the output strictly CFR at the declared fps, which LeRobot requires --
its default ``tolerance_s`` is 1e-4 s, so a frame must sit exactly on i/fps.

Each camera key writes one growing mp4 per (chunk, file); episodes are appended
back to back and located later through ``from_timestamp``/``to_timestamp``.

A camera can also carry frame transforms (see :mod:`hdf52lerobot.imaging`) for
a recorder that wrote frames in the sensor's own colour order or orientation;
those run before the encode, so the mp4, the feature shape and the statistics
all describe the same picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import ConverterConfig
from .imaging import compose, transformed_size
from .logger import get_logger

logger = get_logger("hdf52lerobot.video")


@dataclass
class AppendResult:
    """Where one episode's frames landed inside a camera key's video files."""

    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float
    samples: list[np.ndarray]
    """RGB frames pulled aside for statistics, in ``sample_positions`` order
    (after ``sample_transform`` when one was given)."""


class VideoAggregator:
    """Append episodes of one camera key into rolling, CFR mp4 files."""

    def __init__(
        self,
        root: Path,
        feature: str,
        width: int,
        height: int,
        config: ConverterConfig,
        transform: tuple[str, ...] = (),
    ):
        """
        Args:
            width: Source frame width, as decoded.
            height: Source frame height, as decoded.
            transform: Frame transforms applied before encoding. The encoder is
                sized from the result, so a transform that swaps the axes gives
                an mp4 the other way round from its source.
        """
        self.root = Path(root)
        self.feature = feature
        self.width = int(width)
        self.height = int(height)
        self.config = config
        self.transform = tuple(transform)
        self._apply = compose(self.transform)
        self.output_width, self.output_height = transformed_size(
            self.width, self.height, self.transform
        )

        self._container = None
        self._stream = None
        self._chunk_index = 0
        self._file_index = 0
        self._frames_in_file = 0
        self._frame_time_base = Fraction(1, int(config.fps))

    # -- paths ---------------------------------------------------------------

    def path_for(self, chunk_index: int, file_index: int) -> Path:
        return (
            self.root
            / "videos"
            / self.feature
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )

    @property
    def current_path(self) -> Path:
        return self.path_for(self._chunk_index, self._file_index)

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> None:
        import av

        path = self.current_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(path), "w")
        options = {
            "crf": str(self.config.crf),
            "g": str(self.config.gop),
            "keyint_min": str(self.config.gop),
            "preset": self.config.preset,
        }
        self._stream = self._container.add_stream(
            self.config.vcodec, self.config.fps, options=options
        )
        if self.config.codec_threads:
            # Left to FFmpeg, every encoder sizes itself to the whole machine,
            # and a run with several cameras and several jobs then oversubscribes
            # it many times over. See ConverterConfig.codec_threads.
            self._stream.codec_context.thread_count = self.config.codec_threads
            if self.config.codec_threads == 1:
                self._stream.codec_context.thread_type = "NONE"
        self._stream.pix_fmt = self.config.pix_fmt
        self._stream.width = self.output_width
        self._stream.height = self.output_height

    def _close_current(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        self._stream = None

    def _roll_if_needed(self) -> None:
        """Start a new file once the current one passes the size budget."""
        if self._container is None:
            return
        limit = self.config.video_files_size_in_mb * 1024 * 1024
        path = self.current_path
        if not path.exists() or path.stat().st_size < limit:
            return
        self._close_current()
        self._file_index += 1
        if self._file_index >= self.config.chunks_size:
            self._file_index = 0
            self._chunk_index += 1
        self._frames_in_file = 0

    def close(self) -> None:
        self._close_current()

    # -- writing -------------------------------------------------------------

    def append(
        self,
        source_path: Path,
        indices: np.ndarray,
        sample_positions: Iterable[int] = (),
        sample_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> AppendResult:
        """Encode ``indices`` of ``source_path`` onto the end of the current file.

        Args:
            source_path: The episode's source video.
            indices: Source frame index for every output frame. Must be
                non-decreasing so a single decode pass suffices.
            sample_positions: Output positions to also return as RGB arrays,
                for statistics.
            sample_transform: Applied to each sampled frame as it is captured,
                e.g. a downsample, so the full-resolution frame is released
                immediately instead of being held until the statistics run.

        Returns:
            The file coordinates and timestamp span this episode occupies.
        """
        import av

        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"{source_path.name}: no frames requested")
        if np.any(np.diff(indices) < 0):
            raise ValueError(f"{source_path.name}: frame indices must be non-decreasing")

        self._roll_if_needed()
        if self._container is None:
            self._open()

        wanted = {int(position) for position in sample_positions}
        samples: dict[int, np.ndarray] = {}
        first_frame = self._frames_in_file

        with av.open(str(source_path)) as container:
            if self.config.codec_threads:
                container.streams.video[0].thread_count = self.config.codec_threads
            decoded = container.decode(video=0)
            source_position = -1
            current = None
            for output_position, wanted_index in enumerate(indices):
                while source_position < wanted_index:
                    try:
                        current = next(decoded)
                    except StopIteration as error:
                        raise ValueError(
                            f"{source_path.name}: needs frame {wanted_index} but the stream "
                            f"ended after {source_position + 1} frames"
                        ) from error
                    source_position += 1

                if current.width != self.width or current.height != self.height:
                    raise ValueError(
                        f"{source_path.name}: frame size changed to "
                        f"{current.width}x{current.height}, expected {self.width}x{self.height}"
                    )
                if self._apply is None:
                    frame = current.reformat(format=self.config.pix_fmt)
                    if output_position in wanted:
                        sample = current.to_ndarray(format="rgb24")
                        samples[output_position] = (
                            sample_transform(sample) if sample_transform else sample
                        )
                else:
                    # One decode, one transform: the encoder and the statistics
                    # both take the transformed frame, so they cannot disagree.
                    image = self._apply(current.to_ndarray(format="rgb24"))
                    if output_position in wanted:
                        samples[output_position] = (
                            sample_transform(image) if sample_transform else image
                        )
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(
                        format=self.config.pix_fmt
                    )
                # Explicit CFR timing: the source pts are the recorder's jittery
                # arrival times, and LeRobot needs frame i to sit exactly at i/fps.
                frame.pts = self._frames_in_file
                frame.time_base = self._frame_time_base
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
                self._frames_in_file += 1

        return AppendResult(
            chunk_index=self._chunk_index,
            file_index=self._file_index,
            from_timestamp=first_frame / self.config.fps,
            to_timestamp=self._frames_in_file / self.config.fps,
            samples=[samples[position] for position in sorted(samples)],
        )
