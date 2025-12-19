# src/snake_rl/tools/obs_debug.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def np_summary(x: np.ndarray) -> str:
    try:
        mn = x.min()
        mx = x.max()
    except Exception:
        mn, mx = "?", "?"
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={mn} max={mx}"


def print_tile_grid(grid_hw: np.ndarray, *, max_h: int = 13, max_w: int = 22) -> None:
    """
    Pretty-print a 2D uint tile-id grid as integers (clipped to max_h/max_w).
    """
    if grid_hw.ndim != 2:
        print(f"[obs_debug] expected 2D grid, got shape={tuple(grid_hw.shape)}", flush=True)
        return

    h, w = grid_hw.shape
    hh = min(h, max_h)
    ww = min(w, max_w)
    sub = grid_hw[:hh, :ww]

    width = max(2, len(str(int(sub.max()))) if sub.size else 2)
    fmt = f"{{:>{width}d}}"

    for y in range(hh):
        row = " ".join(fmt.format(int(v)) for v in sub[y])
        print(row, flush=True)

    if hh < h or ww < w:
        print(f"[obs_debug] (clipped print to {hh}x{ww} of full {h}x{w})", flush=True)


def _extract_first_grid_from_array(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to extract a 2D integer grid from common env observation layouts.

    Common cases:
      - (n_envs, C, H, W) -> env0 channel0 -> (H,W)
      - (n_envs, H, W)   -> env0 -> (H,W)
      - (C, H, W)        -> channel0 -> (H,W)
      - (H, W)           -> already grid
    """
    if arr.ndim == 4 and arr.shape[0] >= 1:
        env0 = arr[0]
        if env0.ndim == 3 and env0.shape[0] >= 1:
            g = env0[0]
            return g if g.ndim == 2 else None
        if env0.ndim == 2:
            return env0
        return None

    if arr.ndim == 3 and arr.shape[0] >= 1:
        g = arr[0]
        return g if g.ndim == 2 else None

    if arr.ndim == 2:
        return arr

    return None


def debug_print_obs(
        obs: Any,
        *,
        header: str = "[obs_debug] raw observation:",
        print_grid: bool = True,
        grid_keys: tuple[str, ...] = ("tiles", "tile", "grid", "pixel"),
) -> None:
    """
    Print raw VecEnv observation in a robust way.

    Handles:
      - Box obs: np.ndarray with shape (n_envs, C, H, W) or (n_envs, H, W) etc.
      - Dict obs: dict[str, np.ndarray] values

    If print_grid is enabled, tries to print a small 2D integer grid from:
      - Dict keys listed in grid_keys (default tries 'tiles' first)
      - For array obs, env0 channel0 if it looks like an integer grid
    """
    print(header, flush=True)

    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                print(f"  key={k!r}: {np_summary(v)}", flush=True)
            else:
                print(f"  key={k!r}: type={type(v).__name__}", flush=True)

        if not print_grid:
            return

        for k in grid_keys:
            if k in obs and isinstance(obs[k], np.ndarray):
                arr = obs[k]
                g2 = _extract_first_grid_from_array(arr)
                if g2 is not None and np.issubdtype(g2.dtype, np.integer):
                    print(f"[obs_debug] printing {k!r} grid for env0:", flush=True)
                    print_tile_grid(g2)
                return
        return

    if isinstance(obs, np.ndarray):
        print(f"  {np_summary(obs)}", flush=True)

        if not print_grid:
            return

        g2 = _extract_first_grid_from_array(obs)
        if g2 is not None and np.issubdtype(g2.dtype, np.integer):
            print("[obs_debug] printing env0 channel0 grid:", flush=True)
            print_tile_grid(g2)
        return

    print(f"  type={type(obs).__name__}", flush=True)


__all__ = ["debug_print_obs", "np_summary", "print_tile_grid"]
