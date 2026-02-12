# src/snake_rl/rl/rust_vec_env.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from snake_rl import _core as core
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.obs_utils import (
    pov_pixel_frame,
    pov_tile_frame_with_valid,
)
from snake_rl.envs.view_radius import parse_view_radius
from snake_rl.game.snake_engine import ensure_rust_core, tileset_tile_count, tileset_tile_size
from snake_rl.vocab import load_tile_vocab


class RustVecEnv(VecEnv):
    """
    Rust-backed vectorized env for SB3.

    Note: returns VecEnv-style outputs (obs, rewards, dones, infos).
    """

    def __init__(
        self,
        *,
        env_id: str,
        env_params: dict[str, Any],
        board: Any,
        food_count: int,
        reward: RewardConfig,
        num_envs: int,
        seeds: Iterable[int] | None = None,
    ) -> None:
        self.env_id = str(env_id)
        self.env_params = dict(env_params)
        if not isinstance(board, core.Board):
            raise TypeError("board must be a snake_rl._core.Board instance")
        self.board = board
        self.reward = reward
        self.num_envs = int(num_envs)

        self.tile_size = tileset_tile_size()
        tile_vocab_name = self.env_params.get("tile_vocab")
        self._tile_vocab = load_tile_vocab(tile_vocab_name) if tile_vocab_name is not None else None
        ext = ensure_rust_core()

        seed_list = None
        if seeds is not None:
            seed_list = [int(s) for s in seeds]

        self._vec_game = ext.VecSnakeEngine(
            n=int(self.num_envs),
            board=self.board,
            food_count=int(food_count),
            seeds=seed_list,
        )

        self._max_playable = int(self._vec_game.max_playable_tiles())
        self.max_steps = max(1, int(self._max_playable * float(self.reward.max_steps_factor)))
        self.max_snake_length = int(self._max_playable)
        self.tiny_reward = float(self.reward.step_penalty_scale) / float(self.max_steps)

        self.current_step_since_last_food = np.zeros((self.num_envs,), dtype=np.int32)
        self.initial_snake_length = np.zeros((self.num_envs,), dtype=np.int32)

        self._actions: np.ndarray | None = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = self._make_observation_space()

        super().__init__(self.num_envs, self.observation_space, self.action_space)

    def _make_observation_space(self) -> spaces.Space:
        env_id = self.env_id
        p = self.env_params
        ts = int(self.tile_size)
        h = int(self.board.height)
        w = int(self.board.width)

        if env_id in {"global_pixel", "global_pixel_dir"}:
            remove_border = bool(p.get("remove_border", True))
            ph = h * ts
            pw = w * ts
            if remove_border:
                ph -= 2 * ts
                pw -= 2 * ts
            pixel_space = spaces.Box(low=0, high=255, shape=(1, ph, pw), dtype=np.uint8)
            if env_id == "global_pixel_dir":
                return spaces.Dict({"pixel": pixel_space, "direction": spaces.Discrete(4)})
            return pixel_space

        if env_id in {"pov_pixel", "pov_pixel_fill"}:
            view_radius = p.get("view_radius")
            if view_radius is None:
                raise ValueError("view_radius is required for pov_pixel envs")
            ry, rx = parse_view_radius(view_radius)
            vh = (2 * ry + 1) * ts
            vw = (2 * rx + 1) * ts
            add_oob_mask = bool(p.get("add_oob_mask", False))
            c = 2 if add_oob_mask else 1
            pixel_space = spaces.Box(low=0, high=255, shape=(c, vh, vw), dtype=np.uint8)
            if env_id == "pov_pixel":
                return pixel_space

            fill_bins = p.get("fill_bins")
            fill_space = (
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                if fill_bins is None
                else spaces.Discrete(int(fill_bins))
            )
            return spaces.Dict({"pixel": pixel_space, "fill": fill_space})

        if env_id in {"global_tile_id", "pov_tile_id"}:
            if self._tile_vocab is not None:
                base_num = int(self._tile_vocab.num_classes)
            else:
                base_num = int(tileset_tile_count())

            if env_id == "global_tile_id":
                remove_border = bool(p.get("remove_border", True))
                gh = h - 2 if remove_border else h
                gw = w - 2 if remove_border else w
                return spaces.Box(
                    low=0,
                    high=base_num - 1,
                    shape=(1, gh, gw),
                    dtype=np.uint8,
                )

            view_radius = p.get("view_radius")
            if view_radius is None:
                raise ValueError("view_radius is required for pov_tile_id envs")
            ry, rx = parse_view_radius(view_radius)
            vy = 2 * ry + 1
            vx = 2 * rx + 1
            mask_oob = bool(p.get("mask_oob", False))
            num_classes = base_num + (1 if mask_oob else 0)
            return spaces.Box(
                low=0,
                high=num_classes - 1,
                shape=(1, vy, vx),
                dtype=np.uint8,
            )

        raise ValueError(f"Unsupported env_id for RustVecEnv: {env_id}")

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
        masks = np.asarray(self._vec_game.step(actions), dtype=np.uint32)

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
        env_id = self.env_id
        p = self.env_params
        ts = int(self.tile_size)

        if env_id in {"global_pixel", "global_pixel_dir"}:
            pixels = np.asarray(self._vec_game.pixel_grids(), dtype=np.uint8)
            remove_border = bool(p.get("remove_border", True))
            if remove_border:
                pixels = pixels[:, ts:-ts, ts:-ts]
            pixels = pixels[:, None, :, :].astype(np.uint8, copy=False)
            if env_id == "global_pixel_dir":
                dirs = np.asarray(self._vec_game.directions(), dtype=np.int64)
                return {"pixel": pixels, "direction": dirs}
            return pixels

        if env_id in {"pov_pixel", "pov_pixel_fill"}:
            pixels = np.asarray(self._vec_game.pixel_grids(), dtype=np.uint8)
            view_radius = p.get("view_radius")
            if view_radius is None:
                raise ValueError("view_radius is required for pov_pixel envs")
            rotate_to_head = bool(p.get("rotate_to_head", True))
            add_oob_mask = bool(p.get("add_oob_mask", False))
            pixel_oob_value = int(p.get("pixel_oob_value", 255))
            mask_valid_value = int(p.get("mask_valid_value", 255))
            mask_oob_value = int(p.get("mask_oob_value", 0))

            head_pos = np.asarray(self._vec_game.head_positions(), dtype=np.int32)
            dirs = np.asarray(self._vec_game.directions(), dtype=np.int8)

            frames: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            for i in range(self.num_envs):
                d = _dir_from_i8(int(dirs[i]))
                if d is None:
                    raise RuntimeError("direction is None in rust vec env")
                head = (int(head_pos[i][0]), int(head_pos[i][1]))
                if add_oob_mask:
                    frame, valid = pov_pixel_frame(
                        pixel_grid=pixels[i],
                        tile_size=ts,
                        head=head,
                        direction=d,
                        view_radius=view_radius,
                        rotate_to_head=rotate_to_head,
                        oob_fill_value=pixel_oob_value,
                        return_valid=True,
                    )
                    frames.append(frame)
                    masks.append(valid)
                else:
                    frame = pov_pixel_frame(
                        pixel_grid=pixels[i],
                        tile_size=ts,
                        head=head,
                        direction=d,
                        view_radius=view_radius,
                        rotate_to_head=rotate_to_head,
                        oob_fill_value=pixel_oob_value,
                        return_valid=False,
                    )
                    frames.append(frame)

            frame_arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)
            if not add_oob_mask:
                out = frame_arr[:, None, :, :]
            else:
                mask_arr = np.stack(masks, axis=0)
                mask = np.full(frame_arr.shape, np.uint8(mask_oob_value), dtype=np.uint8)
                mask[mask_arr] = np.uint8(mask_valid_value)
                out = np.stack([frame_arr, mask], axis=1).astype(np.uint8, copy=False)

            if env_id == "pov_pixel":
                return out

            # pov_pixel_fill
            fill_bins = p.get("fill_bins")
            fill = _compute_fill(
                snake_len=np.asarray(self._vec_game.snake_lens(), dtype=np.int32),
                initial_len=self.initial_snake_length,
                max_playable=self.max_snake_length,
                fill_bins=fill_bins,
            )
            return {"pixel": out, "fill": fill}

        if env_id in {"global_tile_id", "pov_tile_id"}:
            grids = np.asarray(self._vec_game.tile_grids(), dtype=np.uint8)
            vocab = self._tile_vocab

            if env_id == "global_tile_id":
                remove_border = bool(p.get("remove_border", True))
                if remove_border:
                    grids = grids[:, 1:-1, 1:-1]
                if vocab is not None:
                    grids = vocab.lut[grids]
                return grids[:, None, :, :].astype(np.uint8, copy=False)

            view_radius = p.get("view_radius")
            if view_radius is None:
                raise ValueError("view_radius is required for pov_tile_id envs")
            rotate_to_head = bool(p.get("rotate_to_head", True))
            mask_oob = bool(p.get("mask_oob", False))

            head_pos = np.asarray(self._vec_game.head_positions(), dtype=np.int32)
            dirs = np.asarray(self._vec_game.directions(), dtype=np.int8)

            frames: list[np.ndarray] = []
            valids: list[np.ndarray] = []
            for i in range(self.num_envs):
                d = _dir_from_i8(int(dirs[i]))
                if d is None:
                    raise RuntimeError("direction is None in rust vec env")
                head = (int(head_pos[i][0]), int(head_pos[i][1]))
                frame, valid = pov_tile_frame_with_valid(
                    tile_grid=grids[i],
                    head=head,
                    direction=d,
                    view_radius=view_radius,
                    rotate_to_head=rotate_to_head,
                )
                frames.append(frame)
                valids.append(valid)

            frame_arr = np.stack(frames, axis=0)
            valid_arr = np.stack(valids, axis=0)

            if vocab is not None:
                frame_arr = vocab.lut[frame_arr]

            if not mask_oob:
                return frame_arr[:, None, :, :].astype(np.uint8, copy=False)

            out = np.zeros_like(frame_arr, dtype=np.uint8)
            out[valid_arr] = (frame_arr[valid_arr].astype(np.uint16) + 1).astype(
                np.uint8, copy=False
            )
            return out[:, None, :, :].astype(np.uint8, copy=False)

        raise ValueError(f"Unsupported env_id for RustVecEnv: {env_id}")


def _dir_from_i8(v: int) -> int | None:
    if v < 0:
        return None
    return int(v)


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


def _compute_fill(*, snake_len, initial_len, max_playable: int, fill_bins: int | None):
    denom = np.maximum(1, (int(max_playable) - initial_len))
    x = (snake_len - initial_len) / denom.astype(np.float32)
    x = np.clip(x, 0.0, 1.0)

    if fill_bins is None:
        return x.reshape((-1, 1)).astype(np.float32)

    b = np.floor(x * int(fill_bins)).astype(np.int64)
    b = np.minimum(b, int(fill_bins) - 1)
    return b
