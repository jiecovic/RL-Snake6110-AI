# src/snake_rl/config/access.py
from __future__ import annotations

from typing import Any


def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
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
    except Exception:
        return int(default)


def get_run_seed(cfg: Any) -> int:
    return require_int(cfg, "run.seed")


def get_run_num_envs(cfg: Any) -> int:
    return require_int(cfg, "run.num_envs")


def get_env_id(cfg: Any) -> str:
    v = cfg_get(cfg, "env.id", None)
    if v is None:
        raise KeyError("Config missing env.id")
    return str(v)


def get_env_params(cfg: Any) -> dict[str, Any]:
    v = cfg_get(cfg, "env.params", None)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise TypeError(f"env.params must be a dict, got {type(v).__name__}")
    return dict(v)


def get_level_params(cfg: Any) -> dict[str, int]:
    h = cfg_get(cfg, "level.height", None)
    w = cfg_get(cfg, "level.width", None)
    f = cfg_get(cfg, "level.food_count", None)
    if h is None or w is None or f is None:
        raise KeyError("Config missing one of: level.height, level.width, level.food_count")
    return {"height": int(h), "width": int(w), "food_count": int(f)}


def get_frame_stack_n(cfg: Any) -> int:
    return max(1, optional_int(cfg, "observation.frame_stack.n_frames", 1))
