# src/snake_rl/config/access.py
from __future__ import annotations

from typing import Any


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
    except Exception:
        return int(default)


def get_run_seed(cfg: Any) -> int:
    return require_int(cfg, "run.seed")


def get_run_num_envs(cfg: Any) -> int:
    return require_int(cfg, "run.num_envs")


def get_env_engine(cfg: Any) -> str:
    v = cfg_get(cfg, "env.engine", None)
    if v is None:
        v = cfg_get(cfg, "env.params.engine", None)
    if v is None:
        return "python"
    return str(v)


def _legacy_env_id_to_obs(env_id: str) -> dict[str, Any] | None:
    key = str(env_id).strip().lower()
    mapping: dict[str, dict[str, Any]] = {
        "world_pixel": {"kind": "pixel", "view": "world"},
        "world_pixel_dir": {"kind": "pixel", "view": "world", "features": {"direction": True}},
        "head_pixel": {"kind": "pixel", "view": "head"},
        "head_pixel_fill": {
            "kind": "pixel",
            "view": "head",
            "features": {"fill": {"enabled": True}},
        },
        "world_tile_id": {"kind": "tile_id", "view": "world"},
        "head_tile_id": {"kind": "tile_id", "view": "head"},
    }
    return mapping.get(key)


def get_env_obs(cfg: Any) -> dict[str, Any]:
    obs = cfg_get(cfg, "env.obs", None)
    if isinstance(obs, dict):
        return dict(obs)

    obs = cfg_get(cfg, "env.observation", None)
    if isinstance(obs, dict):
        return dict(obs)

    env_id = cfg_get(cfg, "env.id", None)
    if env_id is None:
        raise KeyError("Config missing env.obs (and legacy env.id)")

    mapped = _legacy_env_id_to_obs(str(env_id))
    if mapped is None:
        raise ValueError(f"Unknown legacy env.id={env_id!r}")

    # Merge legacy params into obs.params.
    env_params = cfg_get(cfg, "env.params", None)
    params = dict(env_params) if isinstance(env_params, dict) else {}
    params.pop("engine", None)

    obs_top = cfg_get(cfg, "observation.params", None)
    if isinstance(obs_top, dict):
        merged = dict(obs_top)
        merged.update(params)
        params = merged

    if params:
        mapped = dict(mapped)
        mapped["params"] = params

    # Legacy: fill_bins -> features.fill.bins
    if "params" in mapped and isinstance(mapped.get("params"), dict):
        p = dict(mapped.get("params") or {})
        if "fill_bins" in p:
            fill_bins = p.pop("fill_bins")
            mapped["params"] = p
            feats = dict(mapped.get("features") or {})
            f = feats.get("fill")
            if isinstance(f, dict):
                f2 = dict(f)
                f2.setdefault("enabled", True)
                f2.setdefault("bins", fill_bins)
                feats["fill"] = f2
            elif f is True or f is None:
                feats["fill"] = {"enabled": True, "bins": fill_bins}
            mapped["features"] = feats
    return mapped


def get_env_action(cfg: Any) -> str:
    v = cfg_get(cfg, "env.action.type", None)
    if v is None:
        return "relative"
    return str(v)


def get_board_params(cfg: Any) -> dict[str, int]:
    h = cfg_get(cfg, "board.height", None)
    w = cfg_get(cfg, "board.width", None)
    f = cfg_get(cfg, "board.food_count", None)
    if h is None or w is None or f is None:
        # Back-compat: older configs used `level.*`.
        h = cfg_get(cfg, "level.height", None)
        w = cfg_get(cfg, "level.width", None)
        f = cfg_get(cfg, "level.food_count", None)
    if h is None or w is None or f is None:
        raise KeyError("Config missing one of: board.height, board.width, board.food_count")
    return {"height": int(h), "width": int(w), "food_count": int(f)}


def get_frame_stack_n(cfg: Any) -> int:
    n = optional_int(cfg, "env.frame_stack.n_frames", 1)
    if n > 1:
        return int(n)
    # Back-compat
    return max(1, optional_int(cfg, "observation.frame_stack.n_frames", 1))
