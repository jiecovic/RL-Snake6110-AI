# src/snake_rl/training/reporting.py
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO

from snake_rl.config.schema import TrainConfig
from snake_rl.training.run_paths import RunPaths
from snake_rl.utils.model_params import format_sb3_param_report, format_sb3_param_summary
from snake_rl.utils.paths import relpath


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"


def _try_relpath(value: Any, *, base: Path) -> str:
    if isinstance(value, Path):
        return relpath(value, base=base)
    if isinstance(value, str):
        try:
            return relpath(Path(value), base=base)
        except Exception:
            return value
    return str(value)


def _effective_ppo_init_kwargs(model: PPO) -> dict[str, Any]:
    sig = inspect.signature(PPO.__init__)
    keys = [k for k in sig.parameters.keys() if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        out[k] = getattr(model, k, "<not_exposed>")
    return out


def _log_block(logger, header: str, obj: Any) -> None:
    logger.info(header)
    if obj is None:
        logger.info("  <None>")
        return
    for line in repr(obj).splitlines():
        logger.info(line)


def log_policy_network_detailed(*, model: PPO, logger) -> None:
    policy = getattr(model, "policy", None)
    logger.info("Policy Network (detailed):")
    if policy is None:
        logger.info("  <no model.policy>")
        return

    logger.info(f"  policy_class: {policy.__class__.__name__}")

    feat = getattr(policy, "features_extractor", None)
    if feat is not None:
        _log_block(logger, "  features_extractor:", feat)

    mlp = getattr(policy, "mlp_extractor", None)
    if mlp is not None:
        _log_block(logger, "  mlp_extractor:", mlp)

    an = getattr(policy, "action_net", None)
    if an is not None:
        _log_block(logger, "  action_net:", an)

    vn = getattr(policy, "value_net", None)
    if vn is not None:
        _log_block(logger, "  value_net:", vn)


def log_ppo_params(*, model: PPO, cfg: Any, paths: Any, logger) -> None:
    """
    Log SB3 params and model structure.

    Accepts cfg/paths as Any so this can be reused by CLIs that load the frozen YAML dict,
    without needing TrainConfig/RunPaths.
    """
    repo = getattr(paths, "repo_root", Path.cwd())

    logger.info("PPO effective params (SB3):")
    eff = _effective_ppo_init_kwargs(model)
    for k in sorted(eff.keys()):
        v = eff[k]
        if isinstance(v, float):
            vs = _fmt_float(v)
        elif isinstance(v, int):
            vs = _fmt_int(v)
        elif isinstance(v, (str, Path)) and (("log" in k) or ("path" in k) or k.endswith("_dir")):
            vs = _try_relpath(v, base=repo)
        else:
            vs = str(v)
        logger.info(f"  {k}: {vs}")

    logger.info("Model params (summary):")
    logger.info(format_sb3_param_summary(model))
    logger.info("Model params (detailed):")
    logger.info(format_sb3_param_report(model))

    log_policy_network_detailed(model=model, logger=logger)


def _to_frozen_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
    fe = cfg.model.features_extractor
    d: dict[str, Any] = {
        "run": {
            "name": cfg.run.name,
            "seed": int(cfg.run.seed),
            "num_envs": int(cfg.run.num_envs),
            "total_timesteps": int(cfg.run.total_timesteps),
            "checkpoint_freq": int(cfg.run.checkpoint_freq),
        },
        "level": {
            "height": int(cfg.level.height),
            "width": int(cfg.level.width),
            "food_count": int(cfg.level.food_count),
        },
        "env": {
            "id": str(cfg.env.id),
            "params": dict(cfg.env.params),
        },
        "observation": {
            "params": dict(cfg.observation.params),
            "frame_stack": {
                "n_frames": int(cfg.observation.frame_stack.n_frames),
            },
        },
        "model": {
            "features_extractor": {
                "type": str(fe.type),
                "features_dim": int(fe.features_dim),
                "params": dict(fe.params),
            },
            "net_arch": [int(x) for x in cfg.model.net_arch],
        },
        "ppo": dict(cfg.ppo.params),
        "eval": {
            "intermediate": {
                "enabled": bool(cfg.eval.intermediate.enabled),
                "episodes": int(cfg.eval.intermediate.episodes),
                "deterministic": bool(cfg.eval.intermediate.deterministic),
                "seed_offset": int(cfg.eval.intermediate.seed_offset),
            },
            "final": {
                "enabled": bool(cfg.eval.final.enabled),
                "episodes": int(cfg.eval.final.episodes),
                "deterministic": bool(cfg.eval.final.deterministic),
                "seed_offset": int(cfg.eval.final.seed_offset),
            },
        },
    }

    if cfg.run.resume_checkpoint is not None:
        d["run"]["resume_checkpoint"] = str(cfg.run.resume_checkpoint)

    return d


def save_manifest(*, run_dir: Path, cfg: TrainConfig) -> None:
    """
    Persist ONLY the frozen training configuration.

    config_frozen.yaml is the single source of truth for reproducing a run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_frozen.yaml").write_text(
        yaml.safe_dump(
            _to_frozen_yaml_dict(cfg),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
