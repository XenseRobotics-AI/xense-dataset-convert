"""RGB H.264 without chroma subsampling or quantization."""

from __future__ import annotations

try:
    from lerobot.configs.video import RGBEncoderConfig
except ImportError as error:
    raise ImportError("Lossless XTac video writing requires lerobot>=0.6.1") from error


class LosslessRGBEncoder(RGBEncoderConfig):
    """RGB codec extension, importable by LeRobot's video encoder processes.

    Upstream's codec allowlist omits libx264rgb. Keep its PyAV parameter
    validation, while selecting the encoder which actually preserves RGB.
    """

    def resolve_vcodec(self) -> None:
        self.vcodec = "libx264rgb"

    def get_codec_options(self, encoder_threads=None, as_strings=False):
        options = {"crf": 0, "preset": "ultrafast", "g": 1, "bf": 0,
                   "threads": encoder_threads if encoder_threads is not None else 1}
        return {key: str(value) for key, value in options.items()} if as_strings else options


def rgb_encoder() -> RGBEncoderConfig:
    return LosslessRGBEncoder(pix_fmt="rgb24", crf=0, g=1, preset="ultrafast")
