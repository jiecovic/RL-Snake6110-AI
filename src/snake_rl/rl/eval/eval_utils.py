# src/snake_rl/rl/eval/eval_utils.py
from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from gymnasium import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from snake_rl import _core as core
from snake_rl.config.access import (
    get_board_params,
    get_env_action,
    get_env_obs,
    get_frame_stack_n,
    get_run_vec,
)
from snake_rl.config.schema import RewardConfig
from snake_rl.envs.specs import ActionSpec, ObservationSpec
from snake_rl.rl.envs.factory import make_single_env
from snake_rl.rl.envs.rust_vec_env import RustVecEnv
from snake_rl.rl.eval.metrics import is_win_from_info
from snake_rl.rl.metrics import Metrics
from snake_rl.utils.obs import sanitize_observation


def make_eval_vec_env(*, cfg: Any, seeds: list[int], pixel_key: str = "pixel") -> VecEnv:
    if len(seeds) <= 0:
        raise ValueError("seeds must be non-empty")

    vec_kind = str(get_run_vec(cfg)).lower()
    obs_cfg = dict(get_env_obs(cfg))
    obs_spec = ObservationSpec(
        kind=str(obs_cfg.get("kind")),
        view=str(obs_cfg.get("view")),
        params=dict(obs_cfg.get("params", {})),
        features=dict(obs_cfg.get("features", {})),
    )
    obs_spec.kind_norm()
    obs_spec.view_norm()
    action_spec = ActionSpec(type=str(get_env_action(cfg)))

    n_stack = get_frame_stack_n(cfg)

    if vec_kind == "rust":
        reward = _get_reward_from_cfg(cfg)
        board_cfg = get_board_params(cfg)
        board = core.Board(
            width=int(board_cfg["width"]),
            height=int(board_cfg["height"]),
        )
        vec: VecEnv = RustVecEnv(
            obs=obs_spec,
            action=action_spec,
            board=board,
            food_count=int(board_cfg["food_count"]),
            reward=reward,
            num_envs=len(seeds),
            seeds=seeds,
            frame_stack_n=int(n_stack),
        )
    else:
        env_fns = [make_single_env(cfg=cfg, seed=int(s), frame_stack_n=int(n_stack)) for s in seeds]
        env_fns = cast(list[Callable[[], Env]], env_fns)
        if vec_kind in {"dummy", "subproc"}:
            vec = DummyVecEnv(env_fns)
        else:
            raise ValueError("run.vec must be one of: dummy, subproc, rust")
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
        scores_by_ep = np.full((episodes,), np.nan, dtype=np.float64)

        termination_counts: dict[str, int] = {}

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

                if is_win_from_info(info_i):
                    wins_by_ep[ep_idx] = 1

                tc = info_i.get("termination_cause")
                if isinstance(tc, str) and tc.strip():
                    key = tc.strip()
                    termination_counts[key] = termination_counts.get(key, 0) + 1

                if "final_score" in info_i:
                    with suppress(Exception):
                        scores_by_ep[ep_idx] = float(info_i["final_score"])

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
        Metrics.EPISODES: int(episodes),
        Metrics.DETERMINISTIC: bool(deterministic),
        Metrics.SEED_BASE: int(seed_base),
        Metrics.NUM_ENVS: int(n_envs),
        Metrics.N_FRAMES: int(get_frame_stack_n(cfg)),
        Metrics.EP_RETURN_MEAN: float(r.mean()),
        Metrics.EP_RETURN_STD: float(r.std(ddof=0)),
        Metrics.EP_LENGTH_MEAN: float(lengths.mean()),
        Metrics.EP_LENGTH_STD: float(lengths.std(ddof=0)),
        Metrics.EP_WINS: wins,
        Metrics.EP_WIN_RATE: float(wins / float(episodes)),
        Metrics.ENV_OBS: dict(get_env_obs(cfg)),
        Metrics.ENV_ACTION: str(get_env_action(cfg)),
        Metrics.ENV_ENGINE: "rust",
        Metrics.ENV_VEC: str(get_run_vec(cfg)),
    }

    with suppress(Exception):
        out["env_obs"] = dict(get_env_obs(cfg))

    if termination_counts:
        out["termination_counts"] = dict(sorted(termination_counts.items(), key=lambda kv: kv[0]))
        for cause, count in sorted(termination_counts.items()):
            out[f"{Metrics.TERM_PREFIX}{cause}_count"] = int(count)
            out[f"{Metrics.TERM_PREFIX}{cause}_rate"] = float(count / float(episodes))

    scores_mask = ~np.isnan(scores_by_ep)
    if np.any(scores_mask):
        s = scores_by_ep[scores_mask]
        out[Metrics.EP_SCORE_MEAN] = float(s.mean())
        out[Metrics.EP_SCORE_MIN] = float(s.min())
        out[Metrics.EP_SCORE_MAX] = float(s.max())
        score_rates = s / lengths[scores_mask]
        out[Metrics.STEP_SCORE_RATE] = float(score_rates.mean())

    if np.all(lengths > 0):
        return_rates = r / lengths
        out[Metrics.STEP_RETURN_RATE] = float(return_rates.mean())

    return out
