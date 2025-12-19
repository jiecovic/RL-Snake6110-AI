# src/snake_rl/training/eval_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypedDict, cast

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from snake_rl.envs.registry import get_env_cls
from snake_rl.game.level import EmptyLevel
from snake_rl.game.snakegame import SnakeGame
from snake_rl.training.env_factory import apply_frame_stack
from snake_rl.utils.obs import sanitize_observation


class EpisodeEndInfo(TypedDict, total=False):
    termination_cause: str
    final_score: float


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Read nested config values from either:
      - TrainConfig-like objects (attr access)
      - dict configs (key access)
    path like: "env.id" or "level.height"
    """
    cur: Any = cfg
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _get_env_id(cfg: Any) -> str:
    env_id = _cfg_get(cfg, "env.id", None)
    if env_id is None:
        raise KeyError("Config missing env.id")
    return str(env_id)


def _get_env_params_from_cfg(cfg: Any) -> dict[str, Any]:
    params = _cfg_get(cfg, "env.params", None)
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(f"cfg.env.params must be a dict, got {type(params).__name__}")
    return dict(params)


def _get_level_int(cfg: Any, key: str) -> int:
    v = _cfg_get(cfg, f"level.{key}", None)
    if v is None:
        raise KeyError(f"Config missing level.{key}")
    return int(v)


def _get_n_stack_from_cfg(cfg: Any) -> int:
    n = _cfg_get(cfg, "observation.frame_stack.n_frames", 1)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def _is_win_from_info(info: dict[str, Any]) -> bool:
    for k in ("won", "win", "cleared", "episode_won", "episode_win", "success", "is_success"):
        v = info.get(k)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)

    tc = info.get("termination_cause")
    if isinstance(tc, str):
        s = tc.strip().lower()
        if s in {"win", "won", "cleared", "clear", "success", "goal"}:
            return True

    for k in ("won", "win", "cleared", "success"):
        v = info.get(k)
        if isinstance(v, (int, np.integer)) and int(v) in (0, 1):
            return bool(int(v))

    return False


def _make_single_env_fn(*, cfg: Any, seed: int):
    env_id = _get_env_id(cfg)
    env_cls = get_env_cls(env_id)
    env_params = _get_env_params_from_cfg(cfg)

    height = _get_level_int(cfg, "height")
    width = _get_level_int(cfg, "width")
    food_count = _get_level_int(cfg, "food_count")

    def _init():
        level = EmptyLevel(height=height, width=width)
        game = SnakeGame(level=level, food_count=food_count, seed=int(seed))
        env = env_cls(game, **env_params)  # type: ignore[arg-type]
        env.reset(seed=int(seed))
        return env

    return _init


def make_eval_vec_env(*, cfg: Any, seeds: list[int], pixel_key: str = "pixel") -> VecEnv:
    if len(seeds) <= 0:
        raise ValueError("seeds must be non-empty")

    env_fns = [_make_single_env_fn(cfg=cfg, seed=int(s)) for s in seeds]
    if len(env_fns) == 1:
        vec: VecEnv = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns)

    n_stack = _get_n_stack_from_cfg(cfg)
    vec = apply_frame_stack(vec_env=vec, n_stack=n_stack, pixel_key=str(pixel_key))
    return vec


def _obs_set(obs: Any, idx: int, value: Any) -> Any:
    if isinstance(obs, dict) and isinstance(value, dict):
        out = dict(obs)
        for k, v in value.items():
            arr = out.get(k)
            if isinstance(arr, np.ndarray) and isinstance(v, np.ndarray):
                arr[idx] = v
                out[k] = arr
        return out

    if isinstance(obs, np.ndarray) and isinstance(value, np.ndarray):
        obs[idx] = value
        return obs

    return obs


@dataclass
class _SlotState:
    episode_idx: int
    seed: int
    reward: float = 0.0
    length: int = 0


def evaluate_model(
        *,
        model,
        cfg: Any,
        episodes: int,
        deterministic: bool,
        seed_base: int,
        num_envs: int = 1,
        pixel_key: str = "pixel",
        on_episode: Optional[Callable[[int, int, Optional[float]], None]] = None,
) -> Dict[str, Any]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")

    n_envs = max(1, int(num_envs))
    n_envs = min(n_envs, episodes)

    ep_seeds = [int(seed_base) + ep for ep in range(episodes)]
    slot_seeds = ep_seeds[:n_envs]
    next_ep = n_envs

    vec_env = make_eval_vec_env(cfg=cfg, seeds=slot_seeds, pixel_key=str(pixel_key))
    obs = vec_env.reset()

    slots: list[_SlotState] = [_SlotState(episode_idx=i, seed=slot_seeds[i]) for i in range(n_envs)]

    rewards_by_ep = np.zeros((episodes,), dtype=np.float64)
    lengths_by_ep = np.zeros((episodes,), dtype=np.int64)
    wins_by_ep = np.zeros((episodes,), dtype=np.int64)

    termination_counts: Dict[str, int] = {}
    final_scores: list[float] = []

    finished = 0

    if on_episode is not None:
        for _ in range(n_envs):
            on_episode(finished, episodes, None)

    while finished < episodes:
        obs_for_model = sanitize_observation(obs)
        actions, _ = model.predict(obs_for_model, deterministic=bool(deterministic))

        obs, step_rewards, dones, infos = vec_env.step(actions)

        step_rewards = np.asarray(step_rewards, dtype=np.float64).reshape((n_envs,))
        dones = np.asarray(dones, dtype=bool).reshape((n_envs,))
        infos_list = cast(list[dict], infos)

        for i in range(n_envs):
            s = slots[i]
            if s.episode_idx < 0:
                continue

            s.reward += float(step_rewards[i])
            s.length += 1

            if not dones[i]:
                continue

            info_i = infos_list[i] if i < len(infos_list) and isinstance(infos_list[i], dict) else {}
            ep_idx = int(s.episode_idx)

            rewards_by_ep[ep_idx] = float(s.reward)
            lengths_by_ep[ep_idx] = int(s.length)

            if _is_win_from_info(info_i):
                wins_by_ep[ep_idx] = 1

            tc = info_i.get("termination_cause")
            if isinstance(tc, str) and tc.strip():
                key = tc.strip()
                termination_counts[key] = termination_counts.get(key, 0) + 1

            if "final_score" in info_i:
                try:
                    final_scores.append(float(info_i["final_score"]))
                except Exception:
                    pass

            finished += 1
            if on_episode is not None:
                on_episode(finished, episodes, float(s.reward))

            if next_ep < episodes:
                new_ep_idx = next_ep
                new_seed = ep_seeds[new_ep_idx]
                next_ep += 1

                ret = vec_env.env_method("reset", seed=int(new_seed), indices=i)
                try:
                    obs_i = ret[0][0] if isinstance(ret[0], tuple) else ret[0]
                except Exception:
                    obs_i = ret[0] if ret else None
                if obs_i is not None:
                    obs = _obs_set(obs, i, obs_i)

                slots[i] = _SlotState(episode_idx=new_ep_idx, seed=int(new_seed))

                if on_episode is not None:
                    on_episode(finished, episodes, None)
            else:
                slots[i].episode_idx = -1

    vec_env.close()

    r = rewards_by_ep.astype(np.float64)
    l = lengths_by_ep.astype(np.float64)

    wins = int(wins_by_ep.sum())
    out: Dict[str, Any] = {
        "episodes": int(episodes),
        "deterministic": bool(deterministic),
        "seed_base": int(seed_base),
        "num_envs": int(n_envs),
        "n_frames": int(_get_n_stack_from_cfg(cfg)),
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std(ddof=0)),
        "mean_length": float(l.mean()),
        "std_length": float(l.std(ddof=0)),
        "wins": wins,
        "win_rate": float(wins / float(episodes)),
        "env_id": _get_env_id(cfg),
    }

    try:
        out["env_params"] = dict(_get_env_params_from_cfg(cfg))
    except Exception:
        pass

    if termination_counts:
        out["termination_counts"] = dict(sorted(termination_counts.items(), key=lambda kv: kv[0]))

    if final_scores:
        fs = np.asarray(final_scores, dtype=np.float64)
        out["final_score_mean"] = float(fs.mean())
        out["final_score_min"] = float(fs.min())
        out["final_score_max"] = float(fs.max())

    return out
