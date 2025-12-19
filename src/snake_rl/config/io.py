# src/snake_rl/config/io.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, cast

import yaml

from snake_rl.config.schema import (
    EnvConfig,
    EvalConfig,
    EvalPhaseConfig,
    FeaturesExtractorConfig,
    FrameStackConfig,
    LevelConfig,
    ModelConfig,
    ObservationConfig,
    PPOConfig,
    RunConfig,
    TrainConfig,
)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_int(v: Any, ctx: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Expected int in {ctx}, got {type(v).__name__}: {v!r}") from e


def _as_float(v: Any, ctx: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Expected float in {ctx}, got {type(v).__name__}: {v!r}") from e


def _as_bool(v: Any, ctx: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1", "on"}:
            return True
        if s in {"false", "no", "n", "0", "off"}:
            return False
    raise TypeError(f"Expected bool in {ctx}, got {type(v).__name__}: {v!r}")


def _as_opt_str(v: Any, ctx: str) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v)
    except Exception as e:
        raise TypeError(f"Expected str|None in {ctx}, got {type(v).__name__}: {v!r}") from e
    s = s.strip()
    return s or None


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict, got: {type(data).__name__}")
    return cast(dict[str, Any], data)


def _parse_eval_phase(d: Any, ctx: str, defaults: EvalPhaseConfig) -> EvalPhaseConfig:
    if d is None:
        return defaults
    if not isinstance(d, dict):
        raise TypeError(f"Expected '{ctx}' to be a dict")

    enabled = defaults.enabled if "enabled" not in d else _as_bool(d["enabled"], f"{ctx}.enabled")
    episodes = defaults.episodes if "episodes" not in d else _as_int(d["episodes"], f"{ctx}.episodes")
    deterministic = (
        defaults.deterministic
        if "deterministic" not in d
        else _as_bool(d["deterministic"], f"{ctx}.deterministic")
    )
    seed_offset = (
        defaults.seed_offset
        if "seed_offset" not in d
        else _as_int(d["seed_offset"], f"{ctx}.seed_offset")
    )

    return EvalPhaseConfig(
        enabled=enabled,
        episodes=episodes,
        deterministic=deterministic,
        seed_offset=seed_offset,
    )


def _parse_frame_stack(d: Any, ctx: str) -> FrameStackConfig:
    if d is None:
        return FrameStackConfig()

    if not isinstance(d, dict):
        raise TypeError(f"Expected '{ctx}' to be a dict")

    n_frames = 1 if "n_frames" not in d else _as_int(d["n_frames"], f"{ctx}.n_frames")
    if n_frames <= 0:
        raise ValueError(f"{ctx}.n_frames must be >= 1, got {n_frames}")

    return FrameStackConfig(n_frames=n_frames)


def _parse_env_params(env_d: dict[str, Any]) -> dict[str, Any]:
    params_v = env_d.get("params", {})
    if params_v is None:
        return {}
    if not isinstance(params_v, dict):
        raise TypeError("Expected 'env.params' to be a dict")
    return cast(dict[str, Any], params_v)


def _parse_ppo_params(ppo_d: dict[str, Any]) -> dict[str, Any]:
    """
    Pass-through PPO kwargs.

    We intentionally do NOT validate keys here, because SB3 kwargs evolve and we
    want configs to be forward-compatible. SB3 will raise a clear TypeError if
    an unknown kwarg is provided.
    """
    return dict(ppo_d)


def _parse_features_extractor(model_d: dict[str, Any]) -> FeaturesExtractorConfig:
    """
    Parse model.features_extractor with backward compatibility.

    New schema (preferred):
      model:
        features_extractor:
          type: <str>
          features_dim: <int>
          params: { ... }   # optional

    Old schema (back-compat):
      model:
        features_extractor:
          cnn:
            type: <str>
            features_dim: <int>
            ...extra keys...   # (optional; we collect them into params)
    """
    fe_d = _require(model_d, "features_extractor", "model")
    if not isinstance(fe_d, dict):
        raise TypeError("Expected 'model.features_extractor' to be a dict")

    # Back-compat path: model.features_extractor.cnn
    if "cnn" in fe_d:
        cnn_d = fe_d["cnn"]
        if not isinstance(cnn_d, dict):
            raise TypeError("Expected 'model.features_extractor.cnn' to be a dict")

        fe_type = str(_require(cnn_d, "type", "model.features_extractor.cnn"))
        fe_dim = _as_int(
            _require(cnn_d, "features_dim", "model.features_extractor.cnn"),
            "model.features_extractor.cnn.features_dim",
        )

        # Collect any additional keys (helps migrate configs without breakage).
        params: dict[str, Any] = {
            str(k): v for k, v in cnn_d.items() if str(k) not in {"type", "features_dim"}
        }

        return FeaturesExtractorConfig(type=fe_type, features_dim=fe_dim, params=params)

    # New schema path: model.features_extractor.{type, features_dim, params}
    fe_type = str(_require(fe_d, "type", "model.features_extractor"))
    fe_dim = _as_int(_require(fe_d, "features_dim", "model.features_extractor"), "model.features_extractor.features_dim")

    params_v = fe_d.get("params", {})
    if params_v is None:
        params_v = {}
    if not isinstance(params_v, dict):
        raise TypeError("Expected 'model.features_extractor.params' to be a dict")
    params = cast(dict[str, Any], params_v)

    return FeaturesExtractorConfig(type=fe_type, features_dim=fe_dim, params=dict(params))


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

    env_id = str(_require(env_d, "id", "env"))
    env_params = _parse_env_params(env_d)

    # --- observation ---
    obs_d = _require(data, "observation", "root")
    if not isinstance(obs_d, dict):
        raise TypeError("Expected 'observation' to be a dict")

    # Back-compat: older configs placed env kwargs under observation.params.
    obs_params_v = obs_d.get("params", {})
    if obs_params_v is None:
        obs_params_v = {}
    if not isinstance(obs_params_v, dict):
        raise TypeError("Expected 'observation.params' to be a dict")

    frame_stack = _parse_frame_stack(obs_d.get("frame_stack", None), "observation.frame_stack")

    # Merge back-compat env kwargs into env.params, without overwriting explicit env.params.
    merged_env_params = dict(cast(dict[str, Any], obs_params_v))
    merged_env_params.update(env_params)

    env = EnvConfig(
        id=env_id,
        params=merged_env_params,
    )

    # Keep observation.params for now (back-compat), but it should no longer be used as env kwargs.
    observation = ObservationConfig(
        params=cast(dict[str, Any], obs_params_v),
        frame_stack=frame_stack,
    )

    # --- model ---
    model_d = _require(data, "model", "root")
    if not isinstance(model_d, dict):
        raise TypeError("Expected 'model' to be a dict")

    features_extractor = _parse_features_extractor(model_d)

    net_arch_v = _require(model_d, "net_arch", "model")
    if not isinstance(net_arch_v, list):
        raise TypeError("Expected 'model.net_arch' to be a list of ints")
    net_arch = [_as_int(x, "model.net_arch[i]") for x in net_arch_v]

    model = ModelConfig(
        features_extractor=features_extractor,
        net_arch=net_arch,
    )

    # --- ppo (pass-through) ---
    ppo_v = _require(data, "ppo", "root")
    if not isinstance(ppo_v, dict):
        raise TypeError("Expected 'ppo' to be a dict")
    ppo = PPOConfig(params=_parse_ppo_params(ppo_v))

    # --- eval (optional) ---
    eval_defaults = EvalConfig()
    eval_d = data.get("eval", None)
    if eval_d is None:
        eval_cfg = eval_defaults
    else:
        if not isinstance(eval_d, dict):
            raise TypeError("Expected 'eval' to be a dict")
        intermediate = _parse_eval_phase(
            eval_d.get("intermediate", None),
            "eval.intermediate",
            eval_defaults.intermediate,
        )
        final = _parse_eval_phase(
            eval_d.get("final", None),
            "eval.final",
            eval_defaults.final,
        )
        eval_cfg = EvalConfig(intermediate=intermediate, final=final)

    return TrainConfig(
        run=run,
        level=level,
        env=env,
        observation=observation,
        model=model,
        ppo=ppo,
        eval=eval_cfg,
    )


def load_config_yaml(path: str | Path) -> TrainConfig:
    """
    Load a user-facing YAML config (input schema).

    This must NOT be used to read experiments/<run_id>/config_resolved.json.
    For reruns from a run directory, prefer experiments/<run_id>/config_snapshot.yaml.
    """
    return parse_config(load_yaml(path))


# Backwards-compatible alias (kept for now). Prefer load_config_yaml() in new code.
def load_config(path: str | Path) -> TrainConfig:
    return load_config_yaml(path)


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
        run = replace(run, resume_checkpoint=_as_opt_str(resume_checkpoint, "cli.resume"))

    if run is cfg.run:
        return cfg
    return replace(cfg, run=run)
