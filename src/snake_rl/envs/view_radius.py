# src\snake_rl\envs\view_radius.py

from typing import Any

import numpy as np


def parse_view_radius(v: Any) -> tuple[int, int]:
    """
    Parse POV radius.

    Accepted:
      - int r           -> (r, r)
      - (ry, rx) tuple  -> (ry, rx)
      - [ry, rx] list   -> (ry, rx)  (YAML)

    Returns:
      (ry, rx) with ry,rx >= 0
    """
    if isinstance(v, (int, np.integer)):
        r = int(v)
        if r < 0:
            raise ValueError(f"view_radius must be >= 0, got {r}")
        return (r, r)

    if isinstance(v, (tuple, list)) and len(v) == 2:
        ry = int(v[0])
        rx = int(v[1])
        if ry < 0 or rx < 0:
            raise ValueError(f"view_radius must be >= 0, got {(ry, rx)}")
        return (ry, rx)

    raise TypeError(
        f"view_radius must be an int or a pair (ry, rx) / [ry, rx], got {type(v).__name__}: {v}"
    )
