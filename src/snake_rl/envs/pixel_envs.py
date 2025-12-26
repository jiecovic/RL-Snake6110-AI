# src/snake_rl/envs/pixel_envs.py
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces

from snake_rl.envs.base import BaseSnakeEnv
from snake_rl.envs.pixel_obs import FillFeature, PixelObsEnvBase
from snake_rl.game.snakegame import SnakeGame


def _u8(x: int) -> np.uint8:
    return np.uint8(int(x) & 0xFF)


class GlobalPixelEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: global pixel only.

    Observation: Box (1,H,W) uint8
    Stacking: handled by wrappers (train/eval/watch should see consistent behavior).

    Optional cropping (remove_border) is handled here (env-level), not in SnakeGame.
    This mirrors GlobalTileIdEnv behavior for symbolic obs.
    """

    def __init__(self, game: SnakeGame, *, remove_border: bool = True):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.remove_border = bool(remove_border)

        self.game.reset()
        h, w = self.game.pixel_buffer.shape

        if self.remove_border:
            ts = int(self.game.tileset.tile_size)
            if h <= 2 * ts or w <= 2 * ts:
                raise ValueError(
                    f"remove_border=True requires pixel dims > 2*tile_size; got h={h} w={w} tile_size={ts}"
                )
            h -= 2 * ts
            w -= 2 * ts

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8)

    def _maybe_crop(self, frame: np.ndarray) -> np.ndarray:
        if not self.remove_border:
            return frame
        ts = int(self.game.tileset.tile_size)
        return frame[ts:-ts, ts:-ts]

    def get_obs(self):
        frame = self._global_pixel_frame()
        frame = self._maybe_crop(frame)
        return frame[None, :, :].astype(np.uint8, copy=False)


class GlobalPixelDirectionEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: global pixel + absolute direction.

    Observation: Dict(
      pixel=Box((1,H,W), uint8),
      direction=Discrete(4)
    )

    Stacking: handled by wrappers (stack pixel only, keep direction passthrough).

    Optional cropping (remove_border) is handled here (env-level), not in SnakeGame.
    """

    def __init__(self, game: SnakeGame, *, remove_border: bool = True):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.remove_border = bool(remove_border)

        self.game.reset()
        h, w = self.game.pixel_buffer.shape

        if self.remove_border:
            ts = int(self.game.tileset.tile_size)
            if h <= 2 * ts or w <= 2 * ts:
                raise ValueError(
                    f"remove_border=True requires pixel dims > 2*tile_size; got h={h} w={w} tile_size={ts}"
                )
            h -= 2 * ts
            w -= 2 * ts

        self.observation_space = spaces.Dict(
            {
                "pixel": spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8),
                "direction": spaces.Discrete(4),
            }
        )

    def _maybe_crop(self, frame: np.ndarray) -> np.ndarray:
        if not self.remove_border:
            return frame
        ts = int(self.game.tileset.tile_size)
        return frame[ts:-ts, ts:-ts]

    def get_obs(self):
        frame = self._global_pixel_frame()
        frame = self._maybe_crop(frame)
        pixel = frame[None, :, :].astype(np.uint8, copy=False)

        d = self.game.direction
        assert d is not None, "SnakeGame.direction is None (did you call game.reset()?)"
        direction = int(d.value)

        return {
            "pixel": pixel,
            "direction": np.array(direction, dtype=np.int64),
        }


class PovPixelEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: POV pixel only (centered on head).

    If rotate_to_head=True (default), POV is rotated so forward is UP (egocentric).
    If rotate_to_head=False, POV is world-oriented (allocentric).

    Optional OOB mask channel:
      - add_oob_mask=False: obs is (1,H,W) uint8 (pixel only)
      - add_oob_mask=True : obs is (2,H,W) uint8 where:
            channel0 = pixel
            channel1 = mask (mask_valid_value for in-bounds, mask_oob_value for OOB)

    Notes:
      - pixel_oob_value controls the pixel value used for padded OOB in channel0.
        Recommended default without mask: 255 (OOB looks like wall).
        Recommended default with mask   : 0 or 255 both work; mask disambiguates anyway.
    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            view_radius: int,
            rotate_to_head: bool = True,
            add_oob_mask: bool = False,
            pixel_oob_value: int = 255,
            mask_valid_value: int = 255,
            mask_oob_value: int = 0,
    ):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.view_radius = int(view_radius)
        self.rotate_to_head = bool(rotate_to_head)

        self.add_oob_mask = bool(add_oob_mask)
        self.pixel_oob_value = int(pixel_oob_value)
        self.mask_valid_value = int(mask_valid_value)
        self.mask_oob_value = int(mask_oob_value)

        self.game.reset()

        if self.add_oob_mask:
            frame, _valid = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=True,
            )
            c = 2
        else:
            frame = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=False,
            )
            c = 1

        h, w = frame.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def get_obs(self):
        if not self.add_oob_mask:
            frame = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=False,
            )
            return frame[None, :, :].astype(np.uint8, copy=False)

        frame, valid = self._pov_pixel_frame(
            view_radius=self.view_radius,
            rotate_to_head=self.rotate_to_head,
            oob_fill_value=self.pixel_oob_value,
            return_valid=True,
        )

        mask = np.full(frame.shape, _u8(self.mask_oob_value), dtype=np.uint8)
        mask[valid] = _u8(self.mask_valid_value)

        out = np.stack([frame.astype(np.uint8, copy=False), mask], axis=0)  # (2,H,W)
        return out.astype(np.uint8, copy=False)


class PovPixelFillEnv(BaseSnakeEnv, PixelObsEnvBase):
    """
    Baseline: POV pixel + global fill indicator.

    If rotate_to_head=True (default), POV is rotated so forward is UP (egocentric).
    If rotate_to_head=False, POV is world-oriented (allocentric).

    Observation: Dict(
      pixel=Box((C,H,W), uint8),   where C=1 or 2 depending on add_oob_mask
      fill=Box((1,), float32) OR Discrete(fill_bins)
    )

    Stacking: handled by wrappers (stack pixel only, keep fill passthrough).

    Optional OOB mask channel (same convention as PovPixelEnv).
    """

    def __init__(
            self,
            game: SnakeGame,
            *,
            view_radius: int,
            fill_bins: Optional[int] = None,
            rotate_to_head: bool = True,
            add_oob_mask: bool = False,
            pixel_oob_value: int = 255,
            mask_valid_value: int = 255,
            mask_oob_value: int = 0,
    ):
        BaseSnakeEnv.__init__(self, game)
        PixelObsEnvBase.__init__(self, game)

        self.view_radius = int(view_radius)
        self.rotate_to_head = bool(rotate_to_head)
        self._fill = FillFeature(fill_bins=fill_bins)

        self.add_oob_mask = bool(add_oob_mask)
        self.pixel_oob_value = int(pixel_oob_value)
        self.mask_valid_value = int(mask_valid_value)
        self.mask_oob_value = int(mask_oob_value)

        self.game.reset()

        if self.add_oob_mask:
            frame, _valid = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=True,
            )
            c = 2
        else:
            frame = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=False,
            )
            c = 1

        h, w = frame.shape

        fill_space = (
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            if fill_bins is None
            else spaces.Discrete(int(fill_bins))
        )

        self.observation_space = spaces.Dict(
            {
                "pixel": spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8),
                "fill": fill_space,
            }
        )

    def get_obs(self):
        if not self.add_oob_mask:
            frame = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=False,
            )
            pixel = frame[None, :, :].astype(np.uint8, copy=False)
        else:
            frame, valid = self._pov_pixel_frame(
                view_radius=self.view_radius,
                rotate_to_head=self.rotate_to_head,
                oob_fill_value=self.pixel_oob_value,
                return_valid=True,
            )
            mask = np.full(frame.shape, _u8(self.mask_oob_value), dtype=np.uint8)
            mask[valid] = _u8(self.mask_valid_value)
            pixel = np.stack([frame.astype(np.uint8, copy=False), mask], axis=0).astype(np.uint8, copy=False)

        fill = self._fill.compute(
            snake_len=len(self.game.snake),
            initial_len=self.initial_snake_length,
            max_playable=self.max_snake_length,
        )

        return {
            "pixel": pixel,
            "fill": fill,
        }
