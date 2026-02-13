# src/snake_rl/rl/models/model_factory.py
from __future__ import annotations

import inspect
from contextlib import suppress
from pathlib import Path
from typing import Any

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space

from snake_rl.config.schema import TrainConfig
from snake_rl.rl.models.policy_factory import build_policy_kwargs


def _ensure_str_keys(d: dict[Any, Any]) -> dict[str, Any]:
    # Helps type checkers and avoids accidental non-string YAML keys.
    return {str(k): v for k, v in d.items()}


def _filter_valid_algo_kwargs(model_cls: type, d: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in d.items() if k in valid}


def _coerce_algo_types(d: dict[str, Any]) -> dict[str, Any]:
    # Only coerce known scalar PPO-style kwargs; leave callables/dicts/lists alone.
    float_keys = {
        "learning_rate",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "clip_range",
        "clip_range_vf",
        "max_grad_norm",
        "target_kl",
    }
    int_keys = {
        "n_steps",
        "batch_size",
        "n_epochs",
        "seed",
        "verbose",
        "sde_sample_freq",
    }
    bool_keys = {
        "normalize_advantage",
        "use_sde",
    }

    out: dict[str, Any] = dict(d)
    for k, v in list(out.items()):
        if isinstance(v, str):
            s = v.strip()
            if k in float_keys:
                with suppress(ValueError):
                    out[k] = float(s)
            elif k in int_keys:
                with suppress(ValueError):
                    out[k] = int(s)
            elif k in bool_keys:
                if s.lower() in {"true", "yes", "1", "on"}:
                    out[k] = True
                elif s.lower() in {"false", "no", "0", "off"}:
                    out[k] = False
    return out


def _select_policy(observation_space) -> str | type[MultiInputActorCriticPolicy]:
    # SB3 uses "CnnPolicy" for image-like Box spaces and MultiInput* for Dict.
    # For non-image Box (e.g., categorical grids), prefer MlpPolicy
    # (the feature extractor handles structure).
    if isinstance(observation_space, spaces.Dict):
        return MultiInputActorCriticPolicy
    if isinstance(observation_space, spaces.Box) and is_image_space(
        observation_space,
        check_channels=False,
    ):
        return "CnnPolicy"
    return "MlpPolicy"


def _select_policy_recurrent(observation_space) -> str:
    if isinstance(observation_space, spaces.Dict):
        return "MultiInputLstmPolicy"
    if isinstance(observation_space, spaces.Box) and is_image_space(
        observation_space,
        check_channels=False,
    ):
        return "CnnLstmPolicy"
    return "MlpLstmPolicy"


def _require_recurrent_ppo():
    try:
        from sb3_contrib import RecurrentPPO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sb3-contrib is required for recurrent PPO. Install with: pip install sb3-contrib"
        ) from exc
    return RecurrentPPO


def make_or_load_model(
    *,
    cfg: TrainConfig,
    vec_env,
    tensorboard_log: Path,
    resume_path: Path | None,
) -> Any:
    algo = str(cfg.train.algo.type).strip().lower()
    if algo in {"ppo"}:
        model_cls = PPO
        policy = _select_policy(vec_env.observation_space)
    elif algo in {"recurrent_ppo", "rppo"}:
        model_cls = _require_recurrent_ppo()
        policy = _select_policy_recurrent(vec_env.observation_space)
    else:
        raise NotImplementedError(
            f"Unsupported train.algo.type={algo!r}. Expected 'ppo' or 'recurrent_ppo'."
        )

    if resume_path is not None:
        return model_cls.load(str(resume_path), env=vec_env)

    user_algo_kwargs = _ensure_str_keys(dict(cfg.train.algo.params))
    user_policy_kwargs = {}
    if "policy_kwargs" in user_algo_kwargs:
        raw = user_algo_kwargs.pop("policy_kwargs")
        if isinstance(raw, dict):
            user_policy_kwargs = dict(raw)

    policy_kwargs = build_policy_kwargs(
        cfg=cfg,
        observation_space=vec_env.observation_space,
        extra_policy_kwargs=user_policy_kwargs,
    )

    # Pass-through algo kwargs from YAML (filtered to ctor signature + mild type coercion).
    user_algo_kwargs = _coerce_algo_types(user_algo_kwargs)
    user_algo_kwargs = _filter_valid_algo_kwargs(model_cls, user_algo_kwargs)

    # Default: prefer run.seed as SB3 seed unless user explicitly overrides via ppo.seed.
    # This avoids the confusing "seed: None" in effective SB3 params.
    if "seed" not in user_algo_kwargs:
        user_algo_kwargs["seed"] = int(cfg.run.seed)

    algo_kwargs = {
        "policy": policy,
        "env": vec_env,
        "policy_kwargs": policy_kwargs,
        # Note: SB3 stores this string as-is; we already print it relative in log_ppo_params().
        "tensorboard_log": str(tensorboard_log),
        **user_algo_kwargs,
    }

    return model_cls(**algo_kwargs)
