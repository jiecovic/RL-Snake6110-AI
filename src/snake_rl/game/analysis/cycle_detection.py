# src/snake_rl/game/analysis/cycle_detection.py
from __future__ import annotations

from collections import deque

from snake_rl.game.geometry import Point


def has_head_cycle(recent_heads: deque[Point], *, min_cycle_len: int = 4, min_samples: int = 8) -> bool:
    """
    Detect repeating head-position cycles using a simple verification scheme.

    Args:
        recent_heads: deque of recent head positions.
        min_cycle_len: minimum cycle length to consider.
        min_samples: minimum number of samples required before attempting detection.
    """
    path = list(recent_heads)
    n = len(path)

    if n < max(min_samples, 2 * min_cycle_len):
        return False

    tortoise = 0
    hare = 1
    while hare < n:
        if path[tortoise] == path[hare]:
            cycle_len = hare - tortoise
            if cycle_len < min_cycle_len or hare + cycle_len > n:
                return False

            segment1 = path[hare - cycle_len : hare]
            segment2 = path[hare : hare + cycle_len]
            if segment1 == segment2:
                return True

        tortoise += 1
        hare += 2

    return False
