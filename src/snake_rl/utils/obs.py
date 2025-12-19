# src/snake_rl/utils/obs.py
from __future__ import annotations



from typing import Any

import numpy as np


def sanitize_np(x: Any) -> Any:
    """
    Make numpy inputs safe for downstream torch/SB3 code.

    Fixes:
      - negative strides (e.g., views from flips) -> copy()
      - non-contiguous arrays -> ascontiguousarray()

    For non-numpy inputs, returns x unchanged.
    """
    if isinstance(x, np.ndarray):
        if any(s < 0 for s in x.strides):
            x = x.copy()
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
    return x


def sanitize_observation(obs: Any) -> Any:
    """
    Apply sanitize_np to a VecEnv observation.

    Supports:
      - dict obs: sanitize each value
      - ndarray obs: sanitize directly
      - other: passthrough
    """
    if isinstance(obs, dict):
        return {k: sanitize_np(v) for k, v in obs.items()}
    return sanitize_np(obs)


