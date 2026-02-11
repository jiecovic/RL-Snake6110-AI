# src/snake_rl/envs/pixel_obs.py
from __future__ import annotations

from typing import Literal, overload

import numpy as np

from snake_rl.envs.obs_utils import global_pixel_frame, pov_pixel_frame
from snake_rl.game.snakegame import SnakeGame

Radius = int | tuple[int, int]


class PixelObsEnvBase:
    """
    Small helper base for pixel observations.

    This does NOT implement gym.Env. Concrete envs inherit this alongside BaseSnakeEnv.
    It exists to keep pixel-frame extraction logic (global vs POV) out of each env.
    """

    def __init__(self, game: SnakeGame):
        self.game = game
        self._tilesize = self.game.tileset.tile_size

    def _global_pixel_frame(self) -> np.ndarray:
        """
        Return a single global pixel frame as (H,W) uint8.
        """
        return global_pixel_frame(
            pixel_grid=self.game.pixel_buffer.astype(np.uint8, copy=False),
            tile_size=int(self._tilesize),
            remove_border=False,
        )

    @overload
    def _pov_pixel_frame(
        self,
        *,
        view_radius: Radius,
        rotate_to_head: bool = True,
        oob_fill_value: int = 0,
        return_valid: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def _pov_pixel_frame(
        self,
        *,
        view_radius: Radius,
        rotate_to_head: bool = True,
        oob_fill_value: int = 0,
        return_valid: Literal[False] = False,
    ) -> np.ndarray: ...

    def _pov_pixel_frame(
        self,
        *,
        view_radius: Radius,
        rotate_to_head: bool = True,
        oob_fill_value: int = 0,
        return_valid: bool = False,
    ):
        """
        Return a single POV pixel frame centered on the head.

        view_radius:
          - int r        => square POV (r, r)
          - (ry, rx)     => rectangular POV

        rotate_to_head:
          - True  => egocentric (forward is UP). For rectangular radii, (ry, rx) is
                     in egocentric axes: ry = forward/back radius, rx = left/right radius.
          - False => world-oriented crop (ry vertical, rx horizontal).

        oob_fill_value:
          - Pixel value used for padded OOB regions (uint8-ish int).

        return_valid:
          - If True, also return a boolean valid mask (H,W) where True means
            "in-bounds board pixels".

        Output:
          - frame: (H,W) uint8
          - optionally valid: (H,W) bool
        """
        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"
        return pov_pixel_frame(
            pixel_grid=self.game.pixel_buffer.astype(np.uint8, copy=False),
            tile_size=int(self._tilesize),
            head=self.game.get_head_position(),
            direction=d,
            view_radius=view_radius,
            rotate_to_head=bool(rotate_to_head),
            oob_fill_value=int(oob_fill_value),
            return_valid=bool(return_valid),
        )


class FillFeature:
    """
    Global 'crampedness' / 'fill' feature helper.

    Interpretation:
    - Normalizes current snake length relative to playable tiles.
    - Optionally bins the value into a discrete number of bins.
    """

    def __init__(self, *, fill_bins: int | None = None):
        self.fill_bins = None if fill_bins is None else int(fill_bins)

    def compute(self, *, snake_len: int, initial_len: int, max_playable: int):
        # Normalize "how far we are into filling the board" (0..1)
        denom = max(1, (max_playable - initial_len))
        x = (snake_len - initial_len) / denom
        x = float(np.clip(x, 0.0, 1.0))

        if self.fill_bins is None:
            # Scalar feature as Box(1,) float32 (SB3-friendly)
            return np.array([x], dtype=np.float32)

        # Discrete binned feature (0..bins-1)
        b = int(np.floor(x * self.fill_bins))
        return min(b, self.fill_bins - 1)
