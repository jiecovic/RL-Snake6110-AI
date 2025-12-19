# src/snake_rl/utils/logging.py
from __future__ import annotations

import logging
from typing import Optional


def setup_logger(*, name: str, use_rich: bool, level: str) -> logging.Logger:
    """
    Create a configured logger for snake_rl CLI/training entrypoints.

    - Clears existing handlers (so repeated calls don't duplicate output)
    - Disables propagation (to avoid double-logging via root logger)
    - Uses RichHandler if available and requested; otherwise falls back to StreamHandler
    """
    logger = logging.getLogger(str(name))
    logger.handlers.clear()
    logger.propagate = False

    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(lvl)

    handler: Optional[logging.Handler] = None

    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception:
            handler = None

    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))

    logger.addHandler(handler)
    return logger


__all__ = ["setup_logger"]
