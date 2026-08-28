"""Unified logging configuration.

Falls back to the stdlib when spdlog is unavailable, so the converter still
runs in a bare environment.
"""

import logging

SPDLOG_PATTERN = "[%H:%M:%S] [%n] [%^%l%$] %v"
_STDLIB_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"


class _StdlibAdapter:
    """Expose spdlog's method names (``warn``) on a stdlib logger."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warn(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def critical(self, message: str) -> None:
        self._logger.critical(message)


def get_logger(name: str):
    """Create a logger with the repository's unified format.

    Args:
        name: Logger name.

    Returns:
        A spdlog ``ConsoleLogger`` when spdlog is installed, otherwise a stdlib
        logger wrapped to expose the same ``debug/info/warn/error`` methods.
    """
    try:
        import spdlog
    except ImportError:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_STDLIB_FORMAT, datefmt="%H:%M:%S"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return _StdlibAdapter(logger)

    logger = spdlog.ConsoleLogger(name, colored=True, multithreaded=True)
    logger.set_pattern(SPDLOG_PATTERN)
    logger.set_level(spdlog.LogLevel.INFO)
    return logger
