# src/snake_rl/rl/envs/obs_builder.py
from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl.envs.specs import ObservationSpec


def build_vec_obs(
    *,
    spec: ObservationSpec,
    vec_game: Any,
    tile_vocab_name: str | None,
    tile_size: int,
    max_steps: int,
) -> dict[str, Any] | np.ndarray:
    ts = int(tile_size)

    if spec.kind_norm() == "pixel":
        pixels = np.asarray(vec_game.pixel_grids_stacked(), dtype=np.uint8)
        if spec.view_norm() == "world":
            if spec._remove_border():
                pixels = pixels[:, :, ts:-ts, ts:-ts]
            base = pixels.astype(np.uint8, copy=False)
        else:
            view_radius = spec._view_radius()
            rotate_to_head = spec._rotate_to_head()
            add_oob_mask = spec._add_oob_mask()
            pixel_oob_value = spec._pixel_oob_value()
            mask_valid_value = spec._mask_valid_value()
            mask_oob_value = spec._mask_oob_value()

            if add_oob_mask:
                frame_arr, valid = vec_game.head_pixel_views_stacked(
                    (int(view_radius[0]), int(view_radius[1])),
                    rotate_to_head=rotate_to_head,
                    oob_fill_value=int(pixel_oob_value),
                    return_valid=True,
                )
            else:
                frame_arr = vec_game.head_pixel_views_stacked(
                    (int(view_radius[0]), int(view_radius[1])),
                    rotate_to_head=rotate_to_head,
                    oob_fill_value=int(pixel_oob_value),
                    return_valid=False,
                )
                valid = None

            frame_arr = np.asarray(frame_arr, dtype=np.uint8)
            if not add_oob_mask:
                base = frame_arr
            else:
                mask_arr = np.asarray(valid, dtype=bool)
                mask = np.full(frame_arr.shape, np.uint8(mask_oob_value), dtype=np.uint8)
                mask[mask_arr] = np.uint8(mask_valid_value)
                base = np.concatenate([frame_arr, mask], axis=1).astype(np.uint8, copy=False)

        base_key = "pixel"
    else:
        if tile_vocab_name is None:
            grids = np.asarray(vec_game.tile_grids_stacked(), dtype=np.uint8)
        else:
            grids = np.asarray(
                vec_game.tile_grids_stacked_vocab(str(tile_vocab_name)),
                dtype=np.uint8,
            )

        if spec.view_norm() == "world":
            if spec._remove_border():
                grids = grids[:, :, 1:-1, 1:-1]
            base = grids.astype(np.uint8, copy=False)
        else:
            view_radius = spec._view_radius()
            rotate_to_head = spec._rotate_to_head()
            if tile_vocab_name is None:
                frame_arr = vec_game.head_tile_views_stacked(
                    (int(view_radius[0]), int(view_radius[1])),
                    rotate_to_head=rotate_to_head,
                    empty_id=None,
                )
            else:
                frame_arr = vec_game.head_tile_views_stacked_vocab(
                    (int(view_radius[0]), int(view_radius[1])),
                    rotate_to_head=rotate_to_head,
                    empty_id=None,
                    vocab=str(tile_vocab_name),
                )

            frame_arr = np.asarray(frame_arr, dtype=np.uint8)

            base = frame_arr.astype(np.uint8, copy=False)

        base_key = "categorical"

    extras: dict[str, Any] = {}
    if spec._feature_direction():
        extras["direction"] = np.asarray(vec_game.direction_ids(), dtype=np.int64)

    if spec._feature_snake_progress():
        extras["snake_progress"] = np.asarray(
            vec_game.snake_progresses(),
            dtype=np.float32,
        )

    if spec._feature_time_since_food():
        extras["time_since_last_food"] = np.asarray(
            vec_game.time_since_foods_norm(int(max_steps)),
            dtype=np.float32,
        )

    closest_enabled, metric = spec._feature_closest_food()
    if closest_enabled:
        metric_code = {"manhattan": 0, "euclidean": 1, "euclidean_sq": 2}.get(metric, 0)
        arr = np.asarray(vec_game.closest_foods_norm(int(metric_code)), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("closest_foods_norm must return shape (n_envs,3)")
        extras["closest_food_dx"] = arr[:, 0:1]
        extras["closest_food_dy"] = arr[:, 1:2]
        extras["closest_food_dist"] = arr[:, 2:3]

    if (
        spec._feature_collision("collision_ahead")
        or spec._feature_collision("collision_left")
        or spec._feature_collision("collision_right")
    ):
        c_ahead = np.asarray(vec_game.collision_aheads(), dtype=np.float32)
        c_left = np.asarray(vec_game.collision_lefts(), dtype=np.float32)
        c_right = np.asarray(vec_game.collision_rights(), dtype=np.float32)
        if spec._feature_collision("collision_ahead"):
            extras["collision_ahead"] = c_ahead.reshape((-1, 1))
        if spec._feature_collision("collision_left"):
            extras["collision_left"] = c_left.reshape((-1, 1))
        if spec._feature_collision("collision_right"):
            extras["collision_right"] = c_right.reshape((-1, 1))

    if extras:
        return {base_key: base, **extras}
    return base
