# src/snake_rl/training/model_factory.py
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from snake_rl.config.schema import TrainConfig
from snake_rl.training.policy_factory import build_policy_kwargs


def _ensure_str_keys(d: dict[Any, Any]) -> dict[str, Any]:
    # Helps type checkers and avoids accidental non-string YAML keys.
    return {str(k): v for k, v in d.items()}


def _filter_valid_ppo_kwargs(d: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(PPO.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    return {k: v for k, v in d.items() if k in valid}


def _coerce_ppo_types(d: dict[str, Any]) -> dict[str, Any]:
    # Only coerce known scalar PPO kwargs; leave callables/dicts/lists alone.
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
                try:
                    out[k] = float(s)
                except ValueError:
                    pass
            elif k in int_keys:
                try:
                    out[k] = int(s)
                except ValueError:
                    pass
            elif k in bool_keys:
                if s.lower() in {"true", "yes", "1", "on"}:
                    out[k] = True
                elif s.lower() in {"false", "no", "0", "off"}:
                    out[k] = False
    return out


def _select_policy(observation_space) -> str | type[MultiInputActorCriticPolicy]:
    # SB3 uses "CnnPolicy" for Box (images) and MultiInput* for Dict.
    if isinstance(observation_space, spaces.Dict):
        return MultiInputActorCriticPolicy
    return "CnnPolicy"


def make_or_load_model(
        *,
        cfg: TrainConfig,
        vec_env,
        tensorboard_log: Path,
        resume_path: Optional[Path],
) -> PPO:
    if resume_path is not None:
        return PPO.load(str(resume_path), env=vec_env)

    policy_kwargs = build_policy_kwargs(cfg=cfg, observation_space=vec_env.observation_space)
    policy = _select_policy(vec_env.observation_space)

    # Pass-through PPO kwargs from YAML (filtered to ctor signature + mild type coercion).
    user_ppo_kwargs = _ensure_str_keys(dict(cfg.ppo.params))
    user_ppo_kwargs = _coerce_ppo_types(user_ppo_kwargs)
    user_ppo_kwargs = _filter_valid_ppo_kwargs(user_ppo_kwargs)

    # Default: prefer run.seed as SB3 seed unless user explicitly overrides via ppo.seed.
    # This avoids the confusing "seed: None" in effective SB3 params.
    if "seed" not in user_ppo_kwargs:
        user_ppo_kwargs["seed"] = int(cfg.run.seed)

    ppo_kwargs = {
        "policy": policy,
        "env": vec_env,
        "policy_kwargs": policy_kwargs,
        # Note: SB3 stores this string as-is; we already print it relative in log_ppo_params().
        "tensorboard_log": str(tensorboard_log),
        **user_ppo_kwargs,
    }

    return PPO(**ppo_kwargs)
