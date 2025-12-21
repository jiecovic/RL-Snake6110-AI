# src/snake_rl/game/rendering/pygame/surf.py
from __future__ import annotations

import numpy as np
import pygame


def gray255_to_surface(frame: np.ndarray, *, pixel_size: int) -> pygame.Surface:
    """
    Convert a (H,W) uint8 grayscale frame with values in [0,255]
    into a pygame Surface.

    Assumptions / contract:
    - `frame` is already real pixels (uint8 0..255)
    - No binarization or normalization is performed here
    - Rendering is a pure display operation

    This is the single canonical conversion used by pygame renderers.
    """
    arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
    if arr.ndim != 2:
        raise TypeError(f"expected 2D frame (H,W), got {arr.shape}")

    ps = int(pixel_size)
    if ps <= 0:
        raise ValueError(f"pixel_size must be > 0, got {ps}")

    f = arr.astype(np.uint8, copy=False)

    # Upscale using repeat (faster than kron)
    up = np.repeat(np.repeat(f, ps, axis=0), ps, axis=1)

    # Grayscale -> RGB
    rgb = np.repeat(up[:, :, None], 3, axis=2)  # (H*ps, W*ps, 3)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
