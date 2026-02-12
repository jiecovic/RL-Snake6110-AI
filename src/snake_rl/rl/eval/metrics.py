# src/snake_rl/rl/eval/metrics.py
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class EpisodeEndInfo(TypedDict, total=False):
    termination_cause: str
    final_score: float


def is_win_from_info(info: dict[str, Any]) -> bool:
    for k in ("won", "win", "cleared", "episode_won", "episode_win", "success", "is_success"):
        v = info.get(k)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)

    tc = info.get("termination_cause")
    if isinstance(tc, str):
        s = tc.strip().lower()
        if s in {"win", "won", "cleared", "clear", "success", "goal"}:
            return True

    for k in ("won", "win", "cleared", "success"):
        v = info.get(k)
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            return bool(int(v))

    return False
