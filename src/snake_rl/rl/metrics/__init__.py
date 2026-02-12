# src/snake_rl/rl/metrics/__init__.py
from __future__ import annotations

from typing import Any


class Metrics:
    # Episode-level metrics
    EP_RETURN_MEAN = "episode/return_mean"
    EP_RETURN_STD = "episode/return_std"
    EP_LENGTH_MEAN = "episode/length_mean"
    EP_LENGTH_STD = "episode/length_std"
    EP_SCORE_MEAN = "episode/score_mean"
    EP_SCORE_MIN = "episode/score_min"
    EP_SCORE_MAX = "episode/score_max"
    EP_WINS = "episode/wins"
    EP_WIN_RATE = "episode/win_rate"

    # Step-normalized metrics
    STEP_RETURN_RATE = "step/return_rate"
    STEP_SCORE_RATE = "step/score_rate"

    # Termination metrics prefix
    TERM_PREFIX = "termination/"

    # Eval metadata fields
    EPISODES = "episodes"
    DETERMINISTIC = "deterministic"
    SEED_BASE = "seed_base"
    NUM_ENVS = "num_envs"
    N_FRAMES = "n_frames"
    ENV_OBS = "env_obs"
    ENV_ACTION = "env_action"
    ENV_ENGINE = "env_engine"
    ENV_VEC = "env_vec"


def eval_metric_keys() -> tuple[str, ...]:
    return (
        Metrics.EP_RETURN_MEAN,
        Metrics.EP_RETURN_STD,
        Metrics.EP_LENGTH_MEAN,
        Metrics.EP_LENGTH_STD,
        Metrics.EP_SCORE_MEAN,
        Metrics.EP_SCORE_MIN,
        Metrics.EP_SCORE_MAX,
        Metrics.EP_WINS,
        Metrics.EP_WIN_RATE,
        Metrics.STEP_RETURN_RATE,
        Metrics.STEP_SCORE_RATE,
    )


def train_metric_keys() -> tuple[str, ...]:
    return (
        Metrics.EP_SCORE_MEAN,
        Metrics.EP_SCORE_MIN,
        Metrics.EP_SCORE_MAX,
        Metrics.EP_WINS,
        Metrics.EP_WIN_RATE,
    )


def is_termination_metric(key: str) -> bool:
    return str(key).startswith(Metrics.TERM_PREFIX)


def _get_metrics_group(cfg: Any, group: str) -> Any | None:
    metrics_cfg = cfg.get("metrics") if isinstance(cfg, dict) else getattr(cfg, "metrics", None)
    if metrics_cfg is None:
        return None
    if isinstance(metrics_cfg, dict):
        return metrics_cfg.get(group)
    return getattr(metrics_cfg, group, None)


def resolve_log_keys(cfg: Any, *, group: str, default_keys: tuple[str, ...]) -> set[str]:
    group_cfg = _get_metrics_group(cfg, group)
    if group_cfg is None:
        return set(default_keys)
    keys = None
    if isinstance(group_cfg, dict):
        keys = group_cfg.get("keys")
    else:
        keys = getattr(group_cfg, "keys", None)
    if keys is None:
        return set(default_keys)
    return {str(k) for k in list(keys)}


def resolve_log_termination(cfg: Any, *, group: str, default: bool = True) -> bool:
    group_cfg = _get_metrics_group(cfg, group)
    if group_cfg is None:
        return bool(default)
    val = None
    if isinstance(group_cfg, dict):
        val = group_cfg.get("termination")
    else:
        val = getattr(group_cfg, "termination", None)
    if val is None:
        return bool(default)
    return bool(val)


def format_eval_summary(metrics: dict[str, Any]) -> str:
    mean_r = float(metrics.get(Metrics.EP_RETURN_MEAN, float("nan")))
    std_r = float(metrics.get(Metrics.EP_RETURN_STD, float("nan")))
    mean_l = float(metrics.get(Metrics.EP_LENGTH_MEAN, float("nan")))
    win_rate = float(metrics.get(Metrics.EP_WIN_RATE, 0.0))
    wins = int(metrics.get(Metrics.EP_WINS, 0))
    episodes = int(metrics.get(Metrics.EPISODES, 0))
    num_envs = int(metrics.get(Metrics.NUM_ENVS, 1))
    deterministic = bool(metrics.get(Metrics.DETERMINISTIC, False))

    return (
        f"[eval] mean_reward={mean_r:.6g} std_reward={std_r:.6g} "
        f"mean_len={mean_l:.3f} win_rate={win_rate:.3f} ({wins}/{episodes}) "
        f"num_envs={num_envs} deterministic={deterministic}"
    )


__all__ = [
    "Metrics",
    "eval_metric_keys",
    "format_eval_summary",
    "is_termination_metric",
    "resolve_log_keys",
    "resolve_log_termination",
    "train_metric_keys",
]
