# src/snake_rl/config/io.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from snake_rl.config.schema import (
    CNNConfig,
    EnvConfig,
    FeaturesExtractorConfig,
    ModelConfig,
    ObservationConfig,
    PPOConfig,
    RunConfig,
    TrainConfig,
)


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
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


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict, got: {type(data).__name__}")
    return data


def parse_config(data: Dict[str, Any]) -> TrainConfig:
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
        resume_checkpoint=run_d.get("resume_checkpoint", None),
    )

    # --- env ---
    env_d = _require(data, "env", "root")
    if not isinstance(env_d, dict):
        raise TypeError("Expected 'env' to be a dict")

    env = EnvConfig(
        height=_as_int(_require(env_d, "height", "env"), "env.height"),
        width=_as_int(_require(env_d, "width", "env"), "env.width"),
        food_count=_as_int(_require(env_d, "food_count", "env"), "env.food_count"),
    )

    # --- observation ---
    obs_d = _require(data, "observation", "root")
    if not isinstance(obs_d, dict):
        raise TypeError("Expected 'observation' to be a dict")

    observation = ObservationConfig(
        render_mode=str(_require(obs_d, "render_mode", "observation")),
        n_stack=_as_int(_require(obs_d, "n_stack", "observation"), "observation.n_stack"),
        view_radius=_as_int(_require(obs_d, "view_radius", "observation"), "observation.view_radius"),
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
        output_dim=_as_int(_require(cnn_d, "output_dim", "model.features_extractor.cnn"), "cnn.output_dim"),
    )

    features_extractor = FeaturesExtractorConfig(
        type=str(_require(fe_d, "type", "model.features_extractor")),
        cnn=cnn,
    )

    net_arch_v = _require(model_d, "net_arch", "model")
    if not isinstance(net_arch_v, list) or not all(isinstance(x, (int, float, str)) for x in net_arch_v):
        raise TypeError("Expected 'model.net_arch' to be a list of numbers")
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


def save_resolved_config(cfg_text: str, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cfg_text, encoding="utf-8")
