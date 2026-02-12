# src/snake_rl/rl/envs/rust_vec_env.py
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
from snake_rl.rl.envs.obs_builder import build_vec_obs
from snake_rl.rl.envs.termination import termination_cause


class RustVecEnv(VecEnv):
    """
    Rust-backed vectorized env for SB3.

    Note: returns VecEnv-style outputs (obs, rewards, dones, infos).
    """

    metadata = {"render_modes": []}
    render_mode = None

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
        self.render_mode = None
        if not isinstance(board, core.Board):
            raise TypeError("board must be a snake_rl._core.Board instance")
        self.board = board
        self.reward = reward
        self.num_envs = int(num_envs)
        self.frame_stack_n = max(1, int(frame_stack_n))

        self.tile_size = int(core.tileset_tile_size())
        self._tile_vocab_name = self.obs_spec.tile_vocab_name()
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
                cause = termination_cause(int(masks[i]), bool(truncated[i]))
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
        return build_vec_obs(
            spec=self.obs_spec,
            vec_game=self._vec_game,
            tile_vocab_name=self._tile_vocab_name,
            tile_size=int(self.tile_size),
            max_steps=int(self.max_steps),
        )


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
