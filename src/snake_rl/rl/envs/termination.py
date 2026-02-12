# src/snake_rl/rl/envs/termination.py
from __future__ import annotations

from snake_rl import _core as core


def termination_cause(mask: int, truncated: bool) -> str:
    priority = [
        core.MOVE_WIN,
        core.MOVE_HIT_WALL,
        core.MOVE_HIT_SELF,
        core.MOVE_HIT_BOUNDARY,
        core.MOVE_TIMEOUT,
        core.MOVE_NOT_RUNNING,
    ]
    for r in priority:
        if mask & int(r):
            return _cause_label(int(r))
    if truncated:
        return "timeout"
    return "unknown"


def _cause_label(result: int) -> str:
    labels: dict[int, str] = {
        int(core.MOVE_WIN): "win",
        int(core.MOVE_HIT_WALL): "hit_wall",
        int(core.MOVE_HIT_SELF): "hit_self",
        int(core.MOVE_HIT_BOUNDARY): "hit_boundary",
        int(core.MOVE_NOT_RUNNING): "not_running",
        int(core.MOVE_TIMEOUT): "timeout",
    }
    return labels.get(int(result), "unknown")
