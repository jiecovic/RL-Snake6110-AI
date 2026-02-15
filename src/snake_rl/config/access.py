# src/snake_rl/config/access.py
from __future__ import annotations

from typing import Any

from snake_rl.config.schema import ObservationConfig, RewardConfig


def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Read nested config values from either:
      - TrainConfig-like objects (attr access)
      - dict configs (key access)
    path like: "env.obs.kind" or "board.height"
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


def require_int(cfg: Any, path: str) -> int:
    v = cfg_get(cfg, path, None)
    if v is None:
        raise KeyError(f"Config missing {path}")
    return int(v)


def optional_int(cfg: Any, path: str, default: int) -> int:
    v = cfg_get(cfg, path, None)
    if v is None:
        return int(default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def get_run_seed(cfg: Any) -> int:
    return require_int(cfg, "run.seed")


def get_run_num_envs(cfg: Any) -> int:
    v = cfg_get(cfg, "run.num_envs", None)
    if v is None:
        v = cfg_get(cfg, "env.num_envs", None)
    if v is None:
        raise KeyError("Config missing run.num_envs")
    return int(v)


def get_run_vec(cfg: Any) -> str:
    v = cfg_get(cfg, "run.vec", None)
    if v is None:
        v = cfg_get(cfg, "env.vec", None)
    if v is None:
        return "dummy"
    return str(v)


def get_env_obs(cfg: Any) -> dict[str, Any]:
    obs = cfg_get(cfg, "env.obs", None)
    if isinstance(obs, dict):
        return dict(obs)
    if isinstance(obs, ObservationConfig):
        return {
            "kind": str(obs.kind),
            "view": str(obs.view),
            "params": dict(obs.params),
            "features": dict(obs.features),
        }

    raise KeyError("Config missing env.obs")


def get_reward_config(cfg: Any) -> RewardConfig:
    reward = cfg_get(cfg, "reward", None)
    if reward is None:
        return RewardConfig()
    if isinstance(reward, RewardConfig):
        return reward
    if isinstance(reward, dict):
        return RewardConfig(**reward)
    raise TypeError(f"cfg.reward must be a dict or RewardConfig, got {type(reward).__name__}")


def get_feature_tokens(cfg: Any) -> list[list[str]]:
    params = cfg_get(cfg, "feature_extractor.params", None)
    if not isinstance(params, dict):
        return []
    groups = params.get("feature_tokens")
    if groups is None:
        return []
    out: list[list[str]] = []
    for g in groups:
        items = [str(g)] if isinstance(g, str) else [str(x) for x in g]
        items = [s.strip() for s in items if str(s).strip()]
        if not items:
            raise ValueError("feature_tokens entries must be non-empty")
        out.append(items)
    return out


def get_feature_options(cfg: Any) -> dict[str, Any]:
    params = cfg_get(cfg, "feature_extractor.params", None)
    if not isinstance(params, dict):
        return {}
    opts = params.get("feature_options", {})
    return dict(opts) if isinstance(opts, dict) else {}


def get_env_action(cfg: Any) -> str:
    v = cfg_get(cfg, "env.action.type", None)
    if v is None:
        return "relative"
    return str(v)


def get_board_params(cfg: Any) -> dict[str, Any]:
    h = cfg_get(cfg, "board.height", None)
    w = cfg_get(cfg, "board.width", None)
    f = cfg_get(cfg, "board.food_count", None)
    if h is None or w is None or f is None:
        raise KeyError("Config missing one of: board.height, board.width, board.food_count")
    spawn_random_dir = bool(cfg_get(cfg, "board.spawn_random_dir", False))
    return {
        "height": int(h),
        "width": int(w),
        "food_count": int(f),
        "spawn_random_dir": spawn_random_dir,
    }


def get_frame_stack_n(cfg: Any) -> int:
    n = optional_int(cfg, "env.frame_stack.n_frames", 1)
    if n > 1:
        return int(n)
    return int(n)
