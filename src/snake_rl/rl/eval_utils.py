# src/snake_rl/rl/eval_utils.py
from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from gymnasium import Env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from snake_rl import _core as core
from snake_rl.config.access import get_board_params, get_env_id, get_env_params, get_frame_stack_n
from snake_rl.config.schema import RewardConfig
from snake_rl.rl.env_factory import apply_frame_stack, make_single_env
from snake_rl.rl.rust_vec_env import RustVecEnv
from snake_rl.utils.obs import sanitize_observation


class EpisodeEndInfo(TypedDict, total=False):
    termination_cause: str
    final_score: float


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


def make_eval_vec_env(*, cfg: Any, seeds: list[int], pixel_key: str = "pixel") -> VecEnv:
    if len(seeds) <= 0:
        raise ValueError("seeds must be non-empty")

    env_params = get_env_params(cfg)
    engine = str(env_params.get("engine", "python")).lower()

    if engine == "rust":
        reward = _get_reward_from_cfg(cfg)
        board_cfg = get_board_params(cfg)
        board = core.Board(
            width=int(board_cfg["width"]),
            height=int(board_cfg["height"]),
        )
        vec: VecEnv = RustVecEnv(
            env_id=str(get_env_id(cfg)),
            env_params=dict(env_params),
            board=board,
            food_count=int(board_cfg["food_count"]),
            reward=reward,
            num_envs=len(seeds),
            seeds=seeds,
        )
    else:
        env_fns = [make_single_env(cfg=cfg, seed=int(s)) for s in seeds]
        env_fns = cast(list[Callable[[], Env]], env_fns)
        vec = DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns)

    n_stack = get_frame_stack_n(cfg)
    vec = apply_frame_stack(vec_env=vec, n_stack=n_stack, pixel_key=str(pixel_key))
    return vec


def _get_reward_from_cfg(cfg: Any) -> RewardConfig:
    reward = getattr(cfg, "reward", None) if not isinstance(cfg, dict) else cfg.get("reward")
    if reward is None:
        return RewardConfig()
    if isinstance(reward, RewardConfig):
        return reward
    if isinstance(reward, dict):
        return RewardConfig(**reward)
    raise TypeError(f"cfg.reward must be a dict or RewardConfig, got {type(reward).__name__}")


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
    on_episode: Callable[[int, int, float | None], None] | None = None,
) -> dict[str, Any]:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")

    n_envs = max(1, int(num_envs))
    n_envs = min(n_envs, episodes)

    ss = np.random.SeedSequence(int(seed_base))
    children = ss.spawn(episodes)

    # Convert each child SeedSequence into a single uint32 seed Gym can accept.
    ep_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

    slot_seeds = ep_seeds[:n_envs]
    next_ep = n_envs

    vec_env = make_eval_vec_env(cfg=cfg, seeds=slot_seeds, pixel_key=str(pixel_key))
    try:
        obs = vec_env.reset()

        slots: list[_SlotState] = [
            _SlotState(episode_idx=i, seed=slot_seeds[i]) for i in range(n_envs)
        ]

        rewards_by_ep = np.zeros((episodes,), dtype=np.float64)
        lengths_by_ep = np.zeros((episodes,), dtype=np.int64)
        wins_by_ep = np.zeros((episodes,), dtype=np.int64)

        termination_counts: dict[str, int] = {}
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

                info_i = (
                    infos_list[i] if i < len(infos_list) and isinstance(infos_list[i], dict) else {}
                )
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
                    with suppress(Exception):
                        final_scores.append(float(info_i["final_score"]))

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
    finally:
        vec_env.close()

    r = rewards_by_ep.astype(np.float64)
    lengths = lengths_by_ep.astype(np.float64)

    wins = int(wins_by_ep.sum())
    out: dict[str, Any] = {
        "episodes": int(episodes),
        "deterministic": bool(deterministic),
        "seed_base": int(seed_base),
        "num_envs": int(n_envs),
        "n_frames": int(get_frame_stack_n(cfg)),
        "mean_reward": float(r.mean()),
        "std_reward": float(r.std(ddof=0)),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std(ddof=0)),
        "wins": wins,
        "win_rate": float(wins / float(episodes)),
        "env_id": get_env_id(cfg),
    }

    with suppress(Exception):
        out["env_params"] = dict(get_env_params(cfg))

    if termination_counts:
        out["termination_counts"] = dict(sorted(termination_counts.items(), key=lambda kv: kv[0]))

    if final_scores:
        fs = np.asarray(final_scores, dtype=np.float64)
        out["final_score_mean"] = float(fs.mean())
        out["final_score_min"] = float(fs.min())
        out["final_score_max"] = float(fs.max())

    return out
