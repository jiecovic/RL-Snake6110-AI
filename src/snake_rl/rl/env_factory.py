# src/snake_rl/rl/env_factory.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecEnvWrapper,
    VecFrameStack,
    VecMonitor,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from snake_rl.config.access import cfg_get, get_env_params, get_frame_stack_n, require_int
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.registry import get_env_cls
from snake_rl.game.level import EmptyLevel
from snake_rl.game.snakegame import SnakeGame
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


class DictPixelVecFrameStack(VecEnvWrapper):
    """
    VecEnv wrapper that frame-stacks only obs[pixel_key] for Dict observation spaces.

    Assumptions:
      - underlying venv observation_space is spaces.Dict with key pixel_key
      - obs[pixel_key] is a Box with shape (C,H,W)
      - VecEnv returns numpy arrays (SB3 default)

    Output:
      - obs[pixel_key] becomes (C*n_stack, H, W)
      - other dict keys are left unchanged
    """

    def __init__(self, venv: VecEnv, *, n_stack: int, pixel_key: str = "pixel"):
        self.n_stack = int(n_stack)
        self.pixel_key = str(pixel_key)
        super().__init__(venv)

        if self.n_stack <= 1:
            raise ValueError("DictPixelVecFrameStack requires n_stack > 1")

        if not isinstance(self.venv.observation_space, spaces.Dict):
            raise TypeError("DictPixelVecFrameStack requires Dict observation_space")

        if self.pixel_key not in self.venv.observation_space.spaces:
            raise KeyError(
                f"DictPixelVecFrameStack: missing key {self.pixel_key!r} in observation_space"
            )

        pix_space = self.venv.observation_space.spaces[self.pixel_key]
        if not isinstance(pix_space, spaces.Box):
            raise TypeError(f"DictPixelVecFrameStack: obs[{self.pixel_key!r}] must be a Box")

        if pix_space.shape is None or len(pix_space.shape) != 3:
            raise TypeError(
                f"DictPixelVecFrameStack: expected pixel shape (C,H,W), got {pix_space.shape}"
            )

        c, h, w = (int(pix_space.shape[0]), int(pix_space.shape[1]), int(pix_space.shape[2]))
        stacked_shape = (c * self.n_stack, h, w)

        def _stack_bounds(x: Any) -> Any:
            if np.isscalar(x):
                return x
            arr = np.asarray(x)
            if arr.shape == (c, h, w):
                return np.tile(arr, (self.n_stack, 1, 1))
            return np.broadcast_to(arr, (c * self.n_stack, h, w))

        new_spaces = dict(self.venv.observation_space.spaces)
        box_dtype = (
            pix_space.dtype.type if isinstance(pix_space.dtype, np.dtype) else pix_space.dtype
        )
        new_spaces[self.pixel_key] = spaces.Box(
            low=_stack_bounds(pix_space.low),
            high=_stack_bounds(pix_space.high),
            shape=stacked_shape,
            dtype=box_dtype,
        )
        self.observation_space = spaces.Dict(new_spaces)

        self._buf: np.ndarray | None = None  # (n_envs, C*n_stack, H, W)

    def reset(self):
        obs = self.venv.reset()
        return self._reset_buf(cast(dict[str, Any], obs))

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._update_buf(cast(dict[str, Any], obs), dones)
        return obs, rewards, dones, infos

    def _reset_buf(self, obs: dict):
        pix = obs[self.pixel_key]
        if not isinstance(pix, np.ndarray):
            raise TypeError(f"DictPixelVecFrameStack expects numpy obs, got {type(pix).__name__}")

        if pix.ndim != 4:
            raise TypeError(f"DictPixelVecFrameStack expected pix.ndim==4, got {pix.shape}")

        stacked_pix = np.concatenate([pix] * self.n_stack, axis=1)

        if (
            self._buf is None
            or self._buf.shape != stacked_pix.shape
            or self._buf.dtype != stacked_pix.dtype
        ):
            self._buf = np.zeros_like(stacked_pix)

        self._buf[...] = stacked_pix

        out = dict(obs)
        out[self.pixel_key] = self._buf.copy()
        return out

    def _update_buf(self, obs: dict, dones):
        pix = obs[self.pixel_key]
        if not isinstance(pix, np.ndarray):
            raise TypeError(f"DictPixelVecFrameStack expects numpy obs, got {type(pix).__name__}")

        if self._buf is None:
            return self._reset_buf(obs)

        c = int(pix.shape[1])

        self._buf[:, :-c, :, :] = self._buf[:, c:, :, :]
        self._buf[:, -c:, :, :] = pix

        if dones is not None:
            d = np.asarray(dones, dtype=bool)
            for i in np.where(d)[0]:
                rep = np.concatenate([pix[i : i + 1]] * self.n_stack, axis=1)
                self._buf[i : i + 1, :, :, :] = rep

        out = dict(obs)
        out[self.pixel_key] = self._buf.copy()
        return out


def apply_frame_stack(*, vec_env: VecEnv, n_stack: int, pixel_key: str = "pixel") -> VecEnv:
    """
    Apply frame stacking consistently across CLI + training.

    Rules:
      - n_stack <= 1: return vec_env unchanged
      - Dict obs: stack only pixel_key (leave scalar / discrete features untouched)
      - Box obs: stack along channel dimension (works for pixels and tile-id grids)

    This is intentionally strict and small so watch/train/eval can share it.
    """
    n = int(n_stack)
    if n <= 1:
        return vec_env

    obs_space = vec_env.observation_space
    if isinstance(obs_space, spaces.Dict):
        if pixel_key not in obs_space.spaces:
            raise KeyError(
                f"Dict observation has no {pixel_key!r} key; stacking not implemented for this env."
            )
        return DictPixelVecFrameStack(vec_env, n_stack=n, pixel_key=str(pixel_key))

    # Works for pixel Box AND for tile-id Box (C,H,W).
    return VecFrameStack(vec_env, n_stack=n, channels_order="first")


def make_single_env(*, cfg: Any, seed: int) -> Callable[[], Any]:
    """
    Factory for a single Snake environment instance.

    Seeding model:
      - Gymnasium env.reset(seed=...) passes a deterministic seed into the Rust core.
      - Subsequent resets without a seed continue the Rust RNG stream.
    """
    env_id = cfg_get(cfg, "env.id", None)
    if env_id is None:
        raise KeyError("Config missing env.id")
    env_id = str(env_id)
    env_cls = get_env_cls(env_id)

    def _init():
        # Create static level
        level = EmptyLevel(
            height=require_int(cfg, "level.height"),
            width=require_int(cfg, "level.width"),
        )

        # Create game WITHOUT a seed.
        # RNG will be injected from the env's np_random during reset().
        game = SnakeGame(
            level=level,
            food_count=require_int(cfg, "level.food_count"),
        )

        env_params = get_env_params(cfg)
        env_params.pop("engine", None)
        reward_cfg = _get_reward_from_cfg(cfg)
        if "reward" in env_params:
            raise ValueError("env.params must not contain 'reward'; use top-level reward config.")

        # Construct the Gymnasium environment
        env = env_cls(game, reward=reward_cfg, **env_params)  # type: ignore[call-arg]

        # Seed the environment ONCE at creation time.
        # This initializes env.np_random and, via BaseSnakeEnv.reset(),
        # binds that RNG into the SnakeGame instance.
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

    env_params = get_env_params(cfg)
    engine = str(env_params.get("engine", "python")).lower()
    env_params_clean = dict(env_params)
    env_params_clean.pop("engine", None)

    if engine == "rust":
        level = EmptyLevel(
            height=require_int(cfg, "level.height"),
            width=require_int(cfg, "level.width"),
        )
        reward_cfg = _get_reward_from_cfg(cfg)
        vec_env = RustVecEnv(
            env_id=str(cfg_get(cfg, "env.id")),
            env_params=env_params_clean,
            level=level,
            food_count=require_int(cfg, "level.food_count"),
            reward=reward_cfg,
            num_envs=num_envs,
            seeds=child_seeds,
        )
        vec_env = VecMonitor(vec_env)

        n_stack = get_frame_stack_n(cfg)
        vec_env = apply_frame_stack(
            vec_env=vec_env,
            n_stack=n_stack,
            pixel_key="pixel",
        )
        return vec_env

    env_fns = [make_single_env(cfg=cfg, seed=child_seeds[i]) for i in range(num_envs)]

    # Use DummyVecEnv for single-env runs to avoid subprocess overhead
    if num_envs <= 1:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env: VecEnv = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)

    n_stack = get_frame_stack_n(cfg)
    vec_env = apply_frame_stack(
        vec_env=vec_env,
        n_stack=n_stack,
        pixel_key="pixel",
    )

    return vec_env
