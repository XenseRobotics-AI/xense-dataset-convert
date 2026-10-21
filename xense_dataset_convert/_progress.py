"""Terminal logging and throttled progress for conversion stages."""

from __future__ import annotations

import logging
from time import monotonic


def configure_cli_logging(verbose: bool = False) -> None:
    """Configure command-line output without changing library import behavior."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


class Progress:
    """Log stage boundaries and progress at most once every ten seconds."""

    def __init__(self, logger: logging.Logger, label: str, total: int | None = None,
                 unit: str = "items") -> None:
        self.logger, self.label, self.total, self.unit = logger, label, total, unit
        self.started = self.last_report = monotonic()
        logger.info("%s: starting%s", label, f" ({total:,} {unit})" if total is not None else "")

    def update(self, completed: int, *, done: bool = False) -> None:
        now = monotonic()
        if not done and now - self.last_report < 10:
            return
        elapsed = now - self.started
        amount = f"{completed:,} {self.unit}"
        if self.total:
            amount = f"{completed:,}/{self.total:,} {self.unit} ({completed / self.total:.1%})"
        rate = completed / elapsed if elapsed > 0 else 0
        eta = f", ETA {(self.total - completed) / rate:.1f}s" if self.total and rate and not done else ""
        self.logger.info("%s: %s%s, elapsed %.1fs, %.1f %s/s%s", self.label,
                         "completed, " if done else "", amount, elapsed, rate, self.unit, eta)
        self.last_report = now
