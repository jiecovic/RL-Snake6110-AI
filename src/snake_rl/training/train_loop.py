# src/snake_rl/training/train_loop.py
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths
from snake_rl.utils.model_params import format_sb3_param_report, format_sb3_param_summary


def _setup_logging(*, use_rich: bool, level: str) -> logging.Logger:
    logger = logging.getLogger("snake_rl.train")
    logger.handlers.clear()
    logger.propagate = False

    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            return logger
        except Exception:
            pass

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"


def _fmt_path(value: Any, *, repo_root: Path) -> str:
    if isinstance(value, Path):
        p = value
    elif isinstance(value, str):
        try:
            p = Path(value)
        except Exception:
            return str(value)
    else:
        return str(value)

    try:
        rp = p.resolve()
        rr = repo_root.resolve()
        return str(rp.relative_to(rr))
    except Exception:
        return str(p)


def _effective_ppo_init_kwargs(model: PPO) -> dict[str, Any]:
    sig = inspect.signature(PPO.__init__)
    keys = [k for k in sig.parameters.keys() if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        out[k] = getattr(model, k, "<not_exposed>")
    return out


def _log_block(logger: logging.Logger, header: str, obj: Any) -> None:
    """
    Log a multi-line repr() block in a stable way across Rich + plain logging.
    """
    logger.info(header)
    if obj is None:
        logger.info("  <None>")
        return
    for line in repr(obj).splitlines():
        logger.info(line)


def log_policy_network_detailed(*, model: PPO, logger: logging.Logger) -> None:
    """
    Restore the nice SB3 policy dump you had:

    Policy Network (detailed):
      policy_class: ...
      features_extractor:
      <repr>
      mlp_extractor:
      <repr>
      action_net:
      <repr>
      value_net:
      <repr>
    """
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


def log_ppo_params(
        *,
        model: PPO,
        cfg: TrainConfig,
        paths: RunPaths,
        logger: logging.Logger,
) -> None:
    repo_root = paths.repo_root

    logger.info("PPO effective params (SB3):")
    eff = _effective_ppo_init_kwargs(model)
    for k in sorted(eff.keys()):
        v = eff[k]
        if isinstance(v, float):
            vs = _fmt_float(v)
        elif isinstance(v, int):
            vs = _fmt_int(v)
        elif isinstance(v, (str, Path)) and (("log" in k) or ("path" in k) or k.endswith("_dir")):
            vs = _fmt_path(v, repo_root=repo_root)
        else:
            vs = str(v)
        logger.info(f"  {k}: {vs}")

    logger.info("Model params (summary):")
    logger.info(format_sb3_param_summary(model))
    logger.info("Model params (detailed):")
    logger.info(format_sb3_param_report(model))

    # <-- THIS is what you were missing:
    log_policy_network_detailed(model=model, logger=logger)


def _to_effective_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
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
    }

    if cfg.run.resume_checkpoint is not None:
        d["run"]["resume_checkpoint"] = str(cfg.run.resume_checkpoint)

    d["eval"] = {
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
    }

    return d


def save_manifest(*, run_dir: Path, cfg: TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config_resolved.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (run_dir / "config_effective.yaml").write_text(
        yaml.safe_dump(
            _to_effective_yaml_dict(cfg),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(
        *,
        cfg: TrainConfig,
        resume_override: Optional[str] = None,
        use_rich: bool = True,
        log_level: str = "INFO",
) -> RunPaths:
    logger = _setup_logging(use_rich=use_rich, level=log_level)

    paths = make_run_paths(run_name=str(cfg.run.name))
    save_manifest(run_dir=paths.run_dir, cfg=cfg)

    set_random_seed(int(cfg.run.seed))

    vec_env = make_vec_env(cfg=cfg)

    resume_path: Optional[Path] = None
    resume_value = resume_override if resume_override is not None else cfg.run.resume_checkpoint
    if resume_value:
        resume_path = resolve_resume_arg(str(resume_value), paths.experiments_root)

    model: PPO = make_or_load_model(
        cfg=cfg,
        vec_env=vec_env,
        tensorboard_log=paths.tb_dir,
        resume_path=resume_path,
    )

    logger.info(f"[train] run_id={paths.run_id}")
    logger.info(f"[train] run_dir={paths.run_dir}")
    logger.info(f"[train] tb_dir={paths.tb_dir}")
    logger.info(f"[train] checkpoints={paths.checkpoint_dir}")
    logger.info(f"[train] obs_space={vec_env.observation_space}")
    logger.info(f"[train] action_space={vec_env.action_space}")
    logger.info(f"[train] torch={torch.__version__} cuda={torch.cuda.is_available()}")

    log_ppo_params(model=model, cfg=cfg, paths=paths, logger=logger)

    callbacks = make_callbacks(
        cfg=cfg,
        checkpoint_dir=paths.checkpoint_dir,
    )

    model.learn(
        total_timesteps=int(cfg.run.total_timesteps),
        progress_bar=True,
        callback=callbacks,
        tb_log_name="ppo",
    )

    final_path = paths.checkpoint_dir / "final.zip"
    model.save(str(final_path))

    if bool(cfg.eval.final.enabled):
        seed_base = int(cfg.run.seed) + int(cfg.eval.final.seed_offset)
        metrics = evaluate_model(
            model=model,
            cfg=cfg,
            episodes=int(cfg.eval.final.episodes),
            deterministic=bool(cfg.eval.final.deterministic),
            seed_base=seed_base,
        )
        metrics["phase"] = "final"
        metrics["timesteps"] = int(cfg.run.total_timesteps)

        (paths.run_dir / "eval_final.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _append_jsonl(paths.checkpoint_dir / "eval_history.jsonl", metrics)

        logger.info(
            f"[eval-final] mean_reward={metrics['mean_reward']:.6g} "
            f"std_reward={metrics['std_reward']:.6g} "
            f"mean_len={metrics['mean_length']:.3f}"
        )

    vec_env.close()
    logger.info(f"[train] Done. Final model saved to: {final_path}")
    return paths
