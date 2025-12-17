# src/snake_rl/training/eval_utils.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from snake_rl.envs.registry import ENV_REGISTRY
from snake_rl.game.level import EmptyLevel
from snake_rl.game.snakegame import SnakeGame


def _validate_env_id(env_id: str) -> None:
    if env_id not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}")


def _sanitize_np_array(x):
    # PyTorch can't tensorize numpy arrays with negative strides (e.g. flips/slices views).
    if isinstance(x, np.ndarray):
        if any(s < 0 for s in x.strides):
            x = x.copy()
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
    return x


def sanitize_observation(obs):
    if isinstance(obs, dict):
        return {k: _sanitize_np_array(v) for k, v in obs.items()}
    return _sanitize_np_array(obs)


def make_eval_env(*, cfg: Any, seed: int):
    env_id = str(cfg.env.id)
    _validate_env_id(env_id)

    level = EmptyLevel(height=int(cfg.level.height), width=int(cfg.level.width))
    game = SnakeGame(level=level, food_count=int(cfg.level.food_count), seed=int(seed))

    env_cls = ENV_REGISTRY[env_id]
    env = env_cls(game, **dict(cfg.observation.params))
    env.reset(seed=int(seed))
    return env


def evaluate_model(
        *,
        model,
        cfg: Any,
        episodes: int,
        deterministic: bool,
        seed_base: int,
        on_episode: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")

    rewards: list[float] = []
    lengths: list[int] = []
    termination_counts: Dict[str, int] = {}
    final_scores: list[float] = []

    for ep in range(episodes):
        if on_episode is not None:
            on_episode(ep + 1, episodes, None)

        ep_seed = int(seed_base) + ep
        env = make_eval_env(cfg=cfg, seed=ep_seed)

        obs, _info = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        ep_len = 0
        last_info: Optional[dict] = None

        while not done:
            obs_for_model = sanitize_observation(obs)
            action, _state = model.predict(obs_for_model, deterministic=bool(deterministic))
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            ep_reward += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

        env.close()

        rewards.append(ep_reward)
        lengths.append(ep_len)

        if on_episode is not None:
            on_episode(ep + 1, episodes, float(ep_reward))

        if last_info:
            if "termination_cause" in last_info:
                cause = str(last_info["termination_cause"])
                termination_counts[cause] = termination_counts.get(cause, 0) + 1
            if "final_score" in last_info:
                try:
                    final_scores.append(float(last_info["final_score"]))
                except Exception:
                    pass

    r = np.asarray(rewards, dtype=np.float64)
    l = np.asarray(lengths, dtype=np.float64)

    out: Dict[str, Any] = {
        "episodes": episodes,
        "deterministic": bool(deterministic),
        "seed_base": int(seed_base),
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std(ddof=0)),
        "mean_length": float(l.mean()),
        "std_length": float(l.std(ddof=0)),
    }

    if termination_counts:
        out["termination_counts"] = dict(sorted(termination_counts.items(), key=lambda kv: kv[0]))

    if final_scores:
        fs = np.asarray(final_scores, dtype=np.float64)
        out["final_score_mean"] = float(fs.mean())
        out["final_score_min"] = float(fs.min())
        out["final_score_max"] = float(fs.max())

    try:
        out["env_id"] = str(cfg.env.id)
        out["obs_params"] = dict(cfg.observation.params)
    except Exception:
        pass

    return out
