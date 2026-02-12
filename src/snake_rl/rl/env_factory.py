# src/snake_rl/rl/env_factory.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from snake_rl import _core as core
from snake_rl.config.access import (
    cfg_get,
    get_board_params,
    get_env_action,
    get_env_engine,
    get_env_obs,
    get_frame_stack_n,
    require_int,
)
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.snake_env import SnakeEnv
from snake_rl.envs.specs import ActionSpec, ObservationSpec
from snake_rl.game.snake_engine import SnakeEngine
from snake_rl.rl.rust_vec_env import RustVecEnv


def _get_reward_from_cfg(cfg: Any) -> RewardConfig:
    reward = cfg_get(cfg, "reward", None)
    if reward is None:
        return RewardConfig()
    if isinstance(reward, RewardConfig):
        return reward
    if isinstance(reward, dict):
        return RewardConfig(**reward)
    raise TypeError(f"cfg.reward must be a dict or RewardConfig, got {type(reward).__name__}")


def _obs_spec_from_cfg(cfg: Any) -> ObservationSpec:
    obs_cfg = get_env_obs(cfg)
    if not isinstance(obs_cfg, dict):
        raise TypeError(f"env.obs must be a dict, got {type(obs_cfg).__name__}")
    spec = ObservationSpec(
        kind=str(obs_cfg.get("kind")),
        view=str(obs_cfg.get("view")),
        params=dict(obs_cfg.get("params", {})),
        features=dict(obs_cfg.get("features", {})),
    )
    # Validate early for clearer errors.
    spec.kind_norm()
    spec.view_norm()
    return spec


def _action_spec_from_cfg(cfg: Any) -> ActionSpec:
    return ActionSpec(type=str(get_env_action(cfg)))


def make_single_env(*, cfg: Any, seed: int, frame_stack_n: int) -> Callable[[], Any]:
    """
    Factory for a single Snake environment instance.

    Seeding model:
      - Gymnasium env.reset(seed=...) passes a deterministic seed into the Rust core.
      - Subsequent resets without a seed continue the Rust RNG stream.
    """
    obs_spec = _obs_spec_from_cfg(cfg)
    action_spec = _action_spec_from_cfg(cfg)

    def _init():
        # Create engine WITHOUT a seed.
        # RNG will be injected from the env's np_random during reset().
        board_cfg = get_board_params(cfg)
        board = core.Board(
            width=int(board_cfg["width"]),
            height=int(board_cfg["height"]),
        )
        game = SnakeEngine(
            board=board,
            food_count=int(board_cfg["food_count"]),
            frame_stack_n=int(frame_stack_n),
        )

        reward_cfg = _get_reward_from_cfg(cfg)
        # Construct the Gymnasium environment
        env = SnakeEnv(
            game,
            obs=obs_spec,
            action=action_spec,
            reward=reward_cfg,
            frame_stack_n=int(frame_stack_n),
        )

        # Seed the environment ONCE at creation time.
        # This initializes env.np_random and, via BaseSnakeEnv.reset(),
        # binds that RNG into the SnakeEngine instance.
        env.reset(seed=int(seed))

        return env

    return _init


def make_vec_env(*, cfg: Any):
    """
    Create a vectorized Snake environment.

    Seeding model:
      - A single master seed is expanded into independent per-env streams
        using numpy.random.SeedSequence.
      - Each environment receives exactly one deterministic seed at creation.
      - Subsequent episode resets continue the Rust RNG stream (no reseeding).
    """
    base_seed = require_int(cfg, "run.seed")
    num_envs = require_int(cfg, "run.num_envs")

    # Derive independent child seeds from a single master seed
    ss = np.random.SeedSequence(base_seed)
    child_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(num_envs)]

    engine = str(get_env_engine(cfg)).lower()
    obs_spec = _obs_spec_from_cfg(cfg)
    action_spec = _action_spec_from_cfg(cfg)
    n_stack = get_frame_stack_n(cfg)

    if engine == "rust":
        reward_cfg = _get_reward_from_cfg(cfg)
        board_cfg = get_board_params(cfg)
        board = core.Board(
            width=int(board_cfg["width"]),
            height=int(board_cfg["height"]),
        )
        vec_env = RustVecEnv(
            obs=obs_spec,
            action=action_spec,
            board=board,
            food_count=int(board_cfg["food_count"]),
            reward=reward_cfg,
            num_envs=num_envs,
            seeds=child_seeds,
            frame_stack_n=int(n_stack),
        )
        vec_env = VecMonitor(vec_env)
        return vec_env

    env_fns = [
        make_single_env(cfg=cfg, seed=child_seeds[i], frame_stack_n=int(n_stack))
        for i in range(num_envs)
    ]

    # Use DummyVecEnv for single-env runs to avoid subprocess overhead
    if num_envs <= 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env: VecEnv = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)

    return vec_env
