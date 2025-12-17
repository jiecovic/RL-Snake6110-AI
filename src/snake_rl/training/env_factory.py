# src/snake_rl/training/env_factory.py
from __future__ import annotations

from typing import Any, Callable

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from snake_rl.envs.registry import ENV_REGISTRY
from snake_rl.game.level import EmptyLevel
from snake_rl.game.snakegame import SnakeGame


def _validate_env_id(env_id: str) -> None:
    if env_id not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}")


def make_single_env(*, cfg: Any, seed: int) -> Callable[[], Any]:
    _validate_env_id(str(cfg.env.id))

    def _init():
        level = EmptyLevel(height=int(cfg.level.height), width=int(cfg.level.width))
        game = SnakeGame(level=level, food_count=int(cfg.level.food_count), seed=int(seed))

        env_cls = ENV_REGISTRY[str(cfg.env.id)]
        env = env_cls(game, **dict(cfg.observation.params))

        # IMPORTANT: Do not add TimeLimit here.
        # Our reward design assumes episodes end only by true environment termination.
        env.reset(seed=int(seed))
        return env

    return _init


def make_vec_env(*, cfg: Any):
    _validate_env_id(str(cfg.env.id))

    base_seed = int(cfg.run.seed)
    num_envs = int(cfg.run.num_envs)

    env_fns = [make_single_env(cfg=cfg, seed=base_seed + i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env
