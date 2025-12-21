# src/snake_rl/utils/obs_render.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from snake_rl.game.tile_types import TileType
from snake_rl.game.tileset import Tileset


def _as_u8_2d_last_frame(arr: np.ndarray) -> np.ndarray:
    """
    arr is expected to be (N, C, H, W); returns (H, W) uint8 of the last stacked frame.
    """
    if arr.ndim != 4:
        raise TypeError(f"expected (N,C,H,W), got {arr.shape}")
    frame = arr[0, -1, :, :]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    return frame


def _is_binary01_u8(frame: np.ndarray) -> bool:
    # Fast-ish binary check. Works for typical 0/1 mask buffers.
    if frame.size == 0:
        return True
    u = np.unique(frame)
    return u.size <= 2 and np.all((u == 0) | (u == 1))


def obs_last_frame_and_kind(
        obs: Any,
        *,
        pixel_key: str = "pixel",
) -> Tuple[np.ndarray, str]:
    """
    Extract the *last* stacked frame from a VecEnv observation.

    Returns:
      (frame2d_uint8, kind)

    kind:
      - "pixel": values in [0,255] representing an actual pixel image
      - "mask01": values in {0,1} (binary mask semantics; matches game.pixel_buffer)
      - "tile_id": values are TileType ids (uint8), not pixels
    """
    # --- Dict obs: {"pixel": (N,C,H,W), ...}
    if isinstance(obs, dict):
        if pixel_key not in obs:
            raise KeyError(f"obs dict missing key {pixel_key!r}")

        arr = obs[pixel_key]
        arr = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
        frame = _as_u8_2d_last_frame(arr)

        # Decide kind using content, not container type.
        if _is_binary01_u8(frame):
            return frame, "mask01"

        vmax = int(frame.max()) if frame.size else 0
        # If max fits in the TileType range, it's very likely tile ids.
        max_tid = max(int(t.value) for t in TileType)
        if vmax <= max_tid:
            return frame, "tile_id"

        return frame, "pixel"

    # --- Box obs: (N,C,H,W)
    arr = obs if isinstance(obs, np.ndarray) else np.asarray(obs)
    frame = _as_u8_2d_last_frame(arr)

    if _is_binary01_u8(frame):
        return frame, "mask01"

    vmax = int(frame.max()) if frame.size else 0
    max_tid = max(int(t.value) for t in TileType)
    kind = "tile_id" if vmax <= max_tid else "pixel"
    return frame, kind


def tile_id_frame_to_pixels(
        tile_ids: np.ndarray,  # (H,W) uint8 of TileType.value
        *,
        tileset: Tileset,
) -> np.ndarray:
    """
    Convert a (H,W) tile-id grid into a (H*td, W*td) uint8 pixel image using the project's Tileset.
    """
    if tile_ids.ndim != 2:
        raise TypeError(f"tile_ids must be 2D (H,W), got {tile_ids.shape}")

    td = int(tileset.tile_size)
    h, w = int(tile_ids.shape[0]), int(tile_ids.shape[1])

    out = np.zeros((h * td, w * td), dtype=np.uint8)

    cache: dict[int, np.ndarray] = {}
    for tt in TileType:
        if tt == TileType.EMPTY:
            continue
        if tt in tileset:
            cache[int(tt.value)] = np.array(tileset[tt], dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            tid = int(tile_ids[y, x])
            if tid == int(TileType.EMPTY.value):
                continue
            tile = cache.get(tid)
            if tile is None:
                continue
            y0, x0 = y * td, x * td
            out[y0: y0 + td, x0: x0 + td] = tile

    return out


def obs_frame_to_pixels(
        obs: Any,
        *,
        tileset: Optional[Tileset] = None,
        pixel_key: str = "pixel",
) -> np.ndarray:
    """
    Extract last frame and convert to pixels if needed.

    - "pixel": returned as-is (uint8 [0..255])
    - "mask01": returned as-is (uint8 {0,1}) — agent-view should render via gray01 pipeline
    - "tile_id": converted via tileset (tileset required)
    """
    frame, kind = obs_last_frame_and_kind(obs, pixel_key=pixel_key)

    if kind in ("pixel", "mask01"):
        return frame

    if tileset is None:
        raise ValueError("tileset is required to render tile-id observations")

    return tile_id_frame_to_pixels(frame, tileset=tileset)
