# src/snake_rl/training/eval_utils.py
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

import numpy as np
from gymnasium import spaces

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


def _get_n_stack_from_cfg(cfg: Any) -> int:
    fs = getattr(cfg, "observation", None)
    fs = getattr(fs, "frame_stack", None) if fs is not None else None
    n = getattr(fs, "n_frames", 1) if fs is not None else 1
    try:
        n = int(n)
    except Exception:
        n = 1
    return max(1, n)


def _get_env_params_from_cfg(cfg: Any) -> dict[str, Any]:
    env = getattr(cfg, "env", None)
    params = getattr(env, "params", None) if env is not None else None
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(f"cfg.env.params must be a dict, got {type(params).__name__}")
    return dict(params)


def _stack_box_bounds(x: Any, *, n_stack: int) -> Any:
    # x can be scalar or array-like (e.g. (C,H,W)).
    if np.isscalar(x):
        return x
    arr = np.asarray(x)
    if arr.ndim != 3:
        return x
    return np.tile(arr, (int(n_stack), 1, 1))


class DictPixelFrameStack:
    """
    Single-env frame stacker that stacks only obs[pixel_key] for Dict observations.

    Keeps other keys passthrough (e.g. direction, fill).
    """

    def __init__(self, *, observation_space: spaces.Dict, n_stack: int, pixel_key: str = "pixel"):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("DictPixelFrameStack requires Dict observation_space")

        if pixel_key not in observation_space.spaces:
            raise KeyError(f"DictPixelFrameStack: missing key {pixel_key!r} in observation_space")

        pix_space = observation_space.spaces[pixel_key]
        if not isinstance(pix_space, spaces.Box):
            raise TypeError(f"DictPixelFrameStack: obs[{pixel_key!r}] must be a Box")

        if pix_space.shape is None or len(pix_space.shape) != 3:
            raise TypeError(f"DictPixelFrameStack: expected pixel shape (C,H,W), got {pix_space.shape}")

        self.n_stack = int(n_stack)
        self.pixel_key = str(pixel_key)

        c, h, w = (int(pix_space.shape[0]), int(pix_space.shape[1]), int(pix_space.shape[2]))
        self._c = c
        self._h = h
        self._w = w

        self._frames: Deque[np.ndarray] = deque(maxlen=self.n_stack)

        # Build stacked observation space (handy for sanity/debug, even if not always used).
        new_spaces = dict(observation_space.spaces)
        new_spaces[self.pixel_key] = spaces.Box(
            low=_stack_box_bounds(pix_space.low, n_stack=self.n_stack),
            high=_stack_box_bounds(pix_space.high, n_stack=self.n_stack),
            shape=(c * self.n_stack, h, w),
            dtype=pix_space.dtype,
        )
        self.observation_space = spaces.Dict(new_spaces)

    def reset_with(self, pix: np.ndarray) -> None:
        self._frames.clear()
        pix = pix.astype(np.uint8, copy=False)
        for _ in range(self.n_stack):
            self._frames.append(pix)

    def push(self, pix: np.ndarray) -> np.ndarray:
        pix = pix.astype(np.uint8, copy=False)
        if len(self._frames) == 0:
            self.reset_with(pix)
        else:
            self._frames.append(pix)
        return np.concatenate(list(self._frames), axis=0)

    def stack_obs(self, obs: dict, *, is_reset: bool = False) -> dict:
        pix = obs[self.pixel_key]
        if not isinstance(pix, np.ndarray):
            raise TypeError(f"DictPixelFrameStack expects numpy obs, got {type(pix).__name__}")
        if pix.ndim != 3:
            raise TypeError(f"DictPixelFrameStack: expected pixel (C,H,W), got {pix.shape}")

        if is_reset:
            self.reset_with(pix)
            stacked_pix = np.concatenate([pix] * self.n_stack, axis=0)
        else:
            stacked_pix = self.push(pix)

        out = dict(obs)
        out[self.pixel_key] = stacked_pix
        return out


def make_eval_env(*, cfg: Any, seed: int):
    env_id = str(cfg.env.id)
    _validate_env_id(env_id)

    level = EmptyLevel(height=int(cfg.level.height), width=int(cfg.level.width))
    game = SnakeGame(level=level, food_count=int(cfg.level.food_count), seed=int(seed))

    env_cls = ENV_REGISTRY[env_id]
    env_params = _get_env_params_from_cfg(cfg)
    env = env_cls(game, **env_params)
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

    n_stack = _get_n_stack_from_cfg(cfg)

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

        # Apply frame stacking for eval too (match train/watch behavior).
        stacker: Optional[DictPixelFrameStack] = None
        if n_stack > 1:
            obs_space = getattr(env, "observation_space", None)
            if isinstance(obs_space, spaces.Dict):
                stacker = DictPixelFrameStack(observation_space=obs_space, n_stack=n_stack, pixel_key="pixel")
                obs = stacker.stack_obs(obs, is_reset=True)
            else:
                # Box-only: keep it simple here: stack along channel axis (C,H,W)->(C*n,H,W)
                if not isinstance(obs, np.ndarray) or obs.ndim != 3:
                    raise TypeError(f"Expected Box obs shaped (C,H,W), got {type(obs).__name__} {getattr(obs, 'shape', None)}")
                obs = np.concatenate([obs] * n_stack, axis=0)

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

            if n_stack > 1:
                if stacker is not None:
                    obs = stacker.stack_obs(obs, is_reset=bool(done))
                else:
                    # Box-only rolling stack: shift-left by C, append current obs.
                    # obs is (C*n_stack,H,W) at this point, current is (C,H,W).
                    current = obs if (isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[0] in (1,)) else obs
                    current = obs if isinstance(obs, np.ndarray) else obs  # safety; overwritten below

                    current = obs  # placeholder (will be overwritten by step's obs)
                    # We need the *new* single-frame from env.step (which is `obs` right now).
                    single = obs
                    if not isinstance(single, np.ndarray) or single.ndim != 3:
                        raise TypeError(f"Expected Box obs shaped (C,H,W), got {type(single).__name__} {getattr(single, 'shape', None)}")

                    # Initialize/maintain a buffer for box stacking
                    # Use a local variable in function scope per-episode:
                    # easiest: store in stacker_buf
                    # (we can't use stacker because Dict-only)
                    if "stacker_buf" not in locals():
                        stacker_buf = np.concatenate([single] * n_stack, axis=0)
                        c = int(single.shape[0])
                    else:
                        stacker_buf[:-c, :, :] = stacker_buf[c:, :, :]
                        stacker_buf[-c:, :, :] = single
                        if done:
                            stacker_buf[:, :, :] = np.concatenate([single] * n_stack, axis=0)

                    obs = stacker_buf

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
        out["env_params"] = dict(_get_env_params_from_cfg(cfg))
        out["n_frames"] = int(_get_n_stack_from_cfg(cfg))
    except Exception:
        pass

    return out
