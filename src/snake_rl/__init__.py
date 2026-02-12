# src/snake_rl/__init__.py
from __future__ import annotations

from typing import Any

try:
    from . import _core as _core  # type: ignore
except Exception:  # pragma: no cover
    _core = None  # type: ignore[assignment]

__all__ = ["_core"]
