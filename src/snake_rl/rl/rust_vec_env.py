# src/snake_rl/rl/rust_vec_env.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from snake_rl import _core as core
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.specs import ActionSpec, ObservationSpec
from snake_rl.game.snake_engine import ensure_rust_core


class RustVecEnv(VecEnv):
    """
    Rust-backed vectorized env for SB3.

    Note: returns VecEnv-style outputs (obs, rewards, dones, infos).
    """

    def __init__(
        self,
        *,
        obs: ObservationSpec,
        action: ActionSpec | None = None,
        board: Any,
        food_count: int,
        reward: RewardConfig,
        num_envs: int,
        seeds: Iterable[int] | None = None,
        frame_stack_n: int = 1,
    ) -> None:
        self.obs_spec = obs
        self.action_spec = action if action is not None else ActionSpec()
        if not isinstance(board, core.Board):
            raise TypeError("board must be a snake_rl._core.Board instance")
        self.board = board
        self.reward = reward
        self.num_envs = int(num_envs)
        self.frame_stack_n = max(1, int(frame_stack_n))

        self.tile_size = int(core.tileset_tile_size())
        self._tile_vocab = self.obs_spec.load_tile_vocab()
        ext = ensure_rust_core()

        seed_list = None
        if seeds is not None:
            seed_list = [int(s) for s in seeds]

        self._vec_game = ext.VecSnakeEngine(
            n=int(self.num_envs),
            board=self.board,
            food_count=int(food_count),
            seeds=seed_list,
            frame_stack_n=int(self.frame_stack_n),
        )

        self._max_playable = int(self._vec_game.max_playable_tiles())
        self.max_steps = max(1, int(self._max_playable * float(self.reward.max_steps_factor)))
        self.max_snake_length = int(self._max_playable)
        self.tiny_reward = float(self.reward.step_penalty_scale) / float(self.max_steps)

        self.current_step_since_last_food = np.zeros((self.num_envs,), dtype=np.int32)
        self.initial_snake_length = np.zeros((self.num_envs,), dtype=np.int32)

        self._actions: np.ndarray | None = None

        self.action_space = self.action_spec.action_space()
        self.observation_space = self._make_observation_space()

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    def _make_observation_space(self) -> spaces.Space:
        return self.obs_spec.make_space(
            width=int(self.board.width),
            height=int(self.board.height),
            tile_size=int(self.tile_size),
            tile_vocab=self._tile_vocab,
            frame_stack_n=int(self.frame_stack_n),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            ss = np.random.SeedSequence(int(seed))
            seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(self.num_envs)]
        else:
            seeds = None

        self._vec_game.reset(seeds=seeds)
        self.current_step_since_last_food.fill(0)
        self.initial_snake_length = np.asarray(self._vec_game.snake_lens(), dtype=np.int32)
        obs = self._build_obs()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape((self.num_envs,))

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        actions = [int(a) for a in self._actions.tolist()]
        if self.action_spec.kind() == "relative":
            masks = np.asarray(self._vec_game.step(actions), dtype=np.uint32)
        else:
            masks = np.asarray(self._vec_game.step_cardinal(actions), dtype=np.uint32)

        obs = self._build_obs()

        self.current_step_since_last_food += 1
        is_win = (masks & int(core.MOVE_WIN)) > 0
        is_food = (masks & int(core.MOVE_FOOD)) > 0
        is_fatal = (
            masks
            & int(
                core.MOVE_HIT_SELF
                | core.MOVE_HIT_WALL
                | core.MOVE_HIT_BOUNDARY
                | core.MOVE_NOT_RUNNING
                | core.MOVE_WIN
            )
        ) > 0

        reward = np.zeros((self.num_envs,), dtype=np.float32)

        # Win overrides normal per-step penalty
        reward[is_win] += float(self.reward.win_reward)
        reward[~is_win] -= float(self.tiny_reward)

        if np.any(is_food):
            food_bonus = float(self.reward.food_speed_bonus) * (
                1.0 - (self.current_step_since_last_food / float(self.max_steps))
            )
            reward[is_food] += float(self.reward.food_reward)
            reward[is_food] += food_bonus[is_food]
            self.current_step_since_last_food[is_food] = 0

        if np.any(is_fatal):
            reward[is_fatal] -= float(self.reward.fatal_penalty)

        is_truncated = (self.current_step_since_last_food >= int(self.max_steps)) & (~is_fatal)
        if np.any(is_truncated):
            reward[is_truncated] -= float(self.reward.timeout_penalty)

        terminated = is_fatal
        truncated = is_truncated
        dones = terminated | truncated

        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        scores = np.asarray(self._vec_game.scores(), dtype=np.int32)
        for i in range(self.num_envs):
            infos[i]["move_results"] = int(masks[i])
            if dones[i]:
                cause = _termination_cause(int(masks[i]), bool(truncated[i]))
                infos[i]["final_score"] = int(scores[i])
                infos[i]["termination_cause"] = cause

        # Auto-reset done envs (DummyVecEnv behavior)
        if np.any(dones):
            terminal_obs = _index_obs(obs, np.where(dones)[0])
            for i, idx in enumerate(np.where(dones)[0]):
                self._vec_game.reset_one(int(idx), seed=None)
                self.current_step_since_last_food[int(idx)] = 0
                infos[int(idx)]["terminal_observation"] = terminal_obs[i]

            self.initial_snake_length = np.asarray(self._vec_game.snake_lens(), dtype=np.int32)
            obs = self._build_obs()

        return obs, reward, dones, infos

    def close(self) -> None:
        return None

    def seed(self, seed: int | None = None) -> list[int | None]:
        if seed is None:
            return [None for _ in range(self.num_envs)]
        ss = np.random.SeedSequence(int(seed))
        return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(self.num_envs)]

    def get_attr(self, attr_name: str, indices: VecEnvIndices | None = None):
        value = getattr(self, attr_name)
        idxs = _normalize_indices(indices, self.num_envs)
        return [value for _ in idxs]

    def set_attr(self, attr_name: str, value, indices: VecEnvIndices | None = None):
        setattr(self, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices | None = None,
        **method_kwargs,
    ):
        if method_name != "reset":
            raise AttributeError(f"Unsupported env_method: {method_name}")

        idxs = _normalize_indices(indices, self.num_envs)
        results = []
        for i in idxs:
            seed = method_kwargs.get("seed")
            self._vec_game.reset_one(int(i), seed=None if seed is None else int(seed))
            self.current_step_since_last_food[int(i)] = 0
            self.initial_snake_length = np.asarray(self._vec_game.snake_lens(), dtype=np.int32)
            obs = self._build_obs()
            obs_i = _index_obs(obs, np.asarray([int(i)]))[0]
            results.append((obs_i, {}))
        return results

    def render(self, mode: str = "human"):
        return None

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices | None = None):
        idxs = _normalize_indices(indices, self.num_envs)
        return [False for _ in idxs]

    # ---- internal obs building ----

    def _build_obs(self):
        spec = self.obs_spec
        ts = int(self.tile_size)

        if spec.kind_norm() == "pixel":
            pixels = np.asarray(self._vec_game.pixel_grids_stacked(), dtype=np.uint8)
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
                    frame_arr, valid = self._vec_game.head_pixel_views_stacked(
                        (int(view_radius[0]), int(view_radius[1])),
                        rotate_to_head=rotate_to_head,
                        oob_fill_value=int(pixel_oob_value),
                        return_valid=True,
                    )
                else:
                    frame_arr = self._vec_game.head_pixel_views_stacked(
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
            grids = np.asarray(self._vec_game.tile_grids_stacked(), dtype=np.uint8)
            vocab = self._tile_vocab

            if spec.view_norm() == "world":
                if spec._remove_border():
                    grids = grids[:, :, 1:-1, 1:-1]
                if vocab is not None:
                    grids = vocab.lut[grids]
                base = grids.astype(np.uint8, copy=False)
            else:
                view_radius = spec._view_radius()
                rotate_to_head = spec._rotate_to_head()
                frame_arr = self._vec_game.head_tile_views_stacked(
                    (int(view_radius[0]), int(view_radius[1])),
                    rotate_to_head=rotate_to_head,
                    empty_id=None,
                )

                frame_arr = np.asarray(frame_arr, dtype=np.uint8)
                if vocab is not None:
                    frame_arr = vocab.lut[frame_arr]

                base = frame_arr.astype(np.uint8, copy=False)

            base_key = "categorical"

        extras: dict[str, Any] = {}
        if spec._feature_direction():
            extras["direction"] = np.asarray(self._vec_game.direction_ids(), dtype=np.int64)

        if spec._feature_snake_progress():
            extras["snake_progress"] = np.asarray(
                self._vec_game.snake_progresses(),
                dtype=np.float32,
            )

        if spec._feature_time_since_food():
            extras["time_since_last_food"] = np.asarray(
                self._vec_game.time_since_foods_norm(int(self.max_steps)),
                dtype=np.float32,
            )

        closest_enabled, metric = spec._feature_closest_food()
        if closest_enabled:
            metric_code = {"manhattan": 0, "euclidean": 1, "euclidean_sq": 2}.get(metric, 0)
            arr = np.asarray(self._vec_game.closest_foods_norm(int(metric_code)), dtype=np.float32)
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
            c_ahead = np.asarray(self._vec_game.collision_aheads(), dtype=np.float32)
            c_left = np.asarray(self._vec_game.collision_lefts(), dtype=np.float32)
            c_right = np.asarray(self._vec_game.collision_rights(), dtype=np.float32)
            if spec._feature_collision("collision_ahead"):
                extras["collision_ahead"] = c_ahead.reshape((-1, 1))
            if spec._feature_collision("collision_left"):
                extras["collision_left"] = c_left.reshape((-1, 1))
            if spec._feature_collision("collision_right"):
                extras["collision_right"] = c_right.reshape((-1, 1))

        if extras:
            return {base_key: base, **extras}
        return base


def _termination_cause(mask: int, truncated: bool) -> str:
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


def _index_obs(obs, indices: np.ndarray):
    if isinstance(obs, dict):
        return [{k: v[i].copy() for k, v in obs.items()} for i in indices.tolist()]
    return [obs[i].copy() for i in indices.tolist()]


def _normalize_indices(indices: VecEnvIndices | None, n: int) -> list[int]:
    if indices is None:
        return list(range(n))
    if isinstance(indices, int):
        return [indices]
    return [int(i) for i in indices]
