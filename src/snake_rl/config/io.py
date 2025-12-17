# src/snake_rl/config/io.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, cast

import yaml

from snake_rl.config.schema import (
    CNNConfig,
    EnvConfig,
    FeaturesExtractorConfig,
    LevelConfig,
    ModelConfig,
    ObservationConfig,
    ObservationType,
    PPOConfig,
    RunConfig,
    TrainConfig,
)

_ALLOWED_OBS_TYPES: set[str] = {"global", "egocentric", "layers"}


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_int(v: Any, ctx: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise TypeError(f"Expected int in {ctx}, got {type(v).__name__}: {v!r}") from e


def _as_float(v: Any, ctx: str) -> float:
    try:
        return float(v)
    except Exception as e:
        raise TypeError(f"Expected float in {ctx}, got {type(v).__name__}: {v!r}") from e


def _as_opt_str(v: Any, ctx: str) -> Optional[str]:
    if v is None:
        return None
    # YAML can give numbers/bools/etc. Normalize aggressively.
    try:
        return str(v)
    except Exception as e:
        raise TypeError(f"Expected str|None in {ctx}, got {type(v).__name__}: {v!r}") from e


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict, got: {type(data).__name__}")
    return cast(dict[str, Any], data)


def parse_config(data: dict[str, Any]) -> TrainConfig:
    # --- run ---
    run_d = _require(data, "run", "root")
    if not isinstance(run_d, dict):
        raise TypeError("Expected 'run' to be a dict")

    run = RunConfig(
        name=str(_require(run_d, "name", "run")),
        seed=_as_int(_require(run_d, "seed", "run"), "run.seed"),
        num_envs=_as_int(_require(run_d, "num_envs", "run"), "run.num_envs"),
        total_timesteps=_as_int(_require(run_d, "total_timesteps", "run"), "run.total_timesteps"),
        checkpoint_freq=_as_int(_require(run_d, "checkpoint_freq", "run"), "run.checkpoint_freq"),
        resume_checkpoint=_as_opt_str(run_d.get("resume_checkpoint", None), "run.resume_checkpoint"),
    )

    # --- level ---
    level_d = _require(data, "level", "root")
    if not isinstance(level_d, dict):
        raise TypeError("Expected 'level' to be a dict")

    level = LevelConfig(
        height=_as_int(_require(level_d, "height", "level"), "level.height"),
        width=_as_int(_require(level_d, "width", "level"), "level.width"),
        food_count=_as_int(_require(level_d, "food_count", "level"), "level.food_count"),
    )

    # --- env ---
    env_d = _require(data, "env", "root")
    if not isinstance(env_d, dict):
        raise TypeError("Expected 'env' to be a dict")

    env = EnvConfig(
        id=str(_require(env_d, "id", "env")),
        max_steps=_as_int(_require(env_d, "max_steps", "env"), "env.max_steps"),
    )

    # --- observation ---
    obs_d = _require(data, "observation", "root")
    if not isinstance(obs_d, dict):
        raise TypeError("Expected 'observation' to be a dict")

    obs_type_raw = str(_require(obs_d, "type", "observation"))
    if obs_type_raw not in _ALLOWED_OBS_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_OBS_TYPES))
        raise ValueError(f"Invalid observation.type={obs_type_raw!r}. Allowed: {allowed}")

    params_v = obs_d.get("params", {})
    if not isinstance(params_v, dict):
        raise TypeError("Expected 'observation.params' to be a dict")

    observation = ObservationConfig(
        type=cast(ObservationType, obs_type_raw),
        params=cast(dict[str, Any], params_v),
    )

    # --- model ---
    model_d = _require(data, "model", "root")
    if not isinstance(model_d, dict):
        raise TypeError("Expected 'model' to be a dict")

    fe_d = _require(model_d, "features_extractor", "model")
    if not isinstance(fe_d, dict):
        raise TypeError("Expected 'model.features_extractor' to be a dict")

    cnn_d = _require(fe_d, "cnn", "model.features_extractor")
    if not isinstance(cnn_d, dict):
        raise TypeError("Expected 'model.features_extractor.cnn' to be a dict")

    cnn = CNNConfig(
        type=str(_require(cnn_d, "type", "model.features_extractor.cnn")),
        features_dim=_as_int(_require(cnn_d, "features_dim", "model.features_extractor.cnn"), "cnn.features_dim"),
    )

    features_extractor = FeaturesExtractorConfig(
        type=str(_require(fe_d, "type", "model.features_extractor")),
        cnn=cnn,
    )

    net_arch_v = _require(model_d, "net_arch", "model")
    if not isinstance(net_arch_v, list):
        raise TypeError("Expected 'model.net_arch' to be a list of ints")
    net_arch = [_as_int(x, "model.net_arch[i]") for x in net_arch_v]

    model = ModelConfig(
        features_extractor=features_extractor,
        net_arch=net_arch,
    )

    # --- ppo ---
    ppo_d = _require(data, "ppo", "root")
    if not isinstance(ppo_d, dict):
        raise TypeError("Expected 'ppo' to be a dict")

    ppo = PPOConfig(
        n_steps=_as_int(_require(ppo_d, "n_steps", "ppo"), "ppo.n_steps"),
        batch_size=_as_int(_require(ppo_d, "batch_size", "ppo"), "ppo.batch_size"),
        n_epochs=_as_int(_require(ppo_d, "n_epochs", "ppo"), "ppo.n_epochs"),
        gamma=_as_float(_require(ppo_d, "gamma", "ppo"), "ppo.gamma"),
        ent_coef=_as_float(_require(ppo_d, "ent_coef", "ppo"), "ppo.ent_coef"),
        learning_rate=_as_float(_require(ppo_d, "learning_rate", "ppo"), "ppo.learning_rate"),
        verbose=_as_int(_require(ppo_d, "verbose", "ppo"), "ppo.verbose"),
    )

    return TrainConfig(
        run=run,
        level=level,
        env=env,
        observation=observation,
        model=model,
        ppo=ppo,
    )


def load_config(path: str | Path) -> TrainConfig:
    return parse_config(load_yaml(path))


def apply_overrides(
        cfg: TrainConfig,
        *,
        seed: Optional[int] = None,
        num_envs: Optional[int] = None,
        total_timesteps: Optional[int] = None,
        checkpoint_freq: Optional[int] = None,
        resume_checkpoint: Optional[str] = None,
) -> TrainConfig:
    run = cfg.run
    if seed is not None:
        run = replace(run, seed=int(seed))
    if num_envs is not None:
        run = replace(run, num_envs=int(num_envs))
    if total_timesteps is not None:
        run = replace(run, total_timesteps=int(total_timesteps))
    if checkpoint_freq is not None:
        run = replace(run, checkpoint_freq=int(checkpoint_freq))
    if resume_checkpoint is not None:
        run = replace(run, resume_checkpoint=resume_checkpoint)

    if run is cfg.run:
        return cfg
    return replace(cfg, run=run)
