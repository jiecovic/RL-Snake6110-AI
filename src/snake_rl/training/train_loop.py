# src/snake_rl/training/train_loop.py
from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"


def _fmt_path(value: Any, *, repo_root: Path) -> str:
    # Pretty-print paths relative to repo_root when possible.
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
        rel = rp.relative_to(rr)
        return str(rel)
    except Exception:
        return str(p)


def _effective_ppo_init_kwargs(model: PPO) -> dict[str, Any]:
    """
    Return effective values for all PPO.__init__ parameters, as exposed by the model.

    Note: not every ctor arg is necessarily stored as an attribute in SB3.
    Those show as "<not_exposed>".
    """
    sig = inspect.signature(PPO.__init__)
    keys = [k for k in sig.parameters.keys() if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        if hasattr(model, k):
            out[k] = getattr(model, k)
        else:
            out[k] = "<not_exposed>"
    return out


def log_ppo_params(
        model: PPO,
        cfg: TrainConfig,
        *,
        paths: RunPaths,
        show_policy_network: bool = True,
) -> None:
    console = Console()
    repo_root = paths.repo_root

    console.print("[bold magenta]PPO Effective Params (SB3):[/bold magenta]")
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
        console.print(f"  [green]{k}:[/green] {vs}")

    if show_policy_network:
        policy = model.policy
        console.print("[bold magenta]Policy Network (detailed):[/bold magenta]")
        console.print(f"  [green]policy_class:[/green] {policy.__class__.__name__}")

        if hasattr(policy, "features_extractor"):
            console.print("  [green]features_extractor:[/green]")
            console.print(policy.features_extractor)

        if hasattr(policy, "mlp_extractor"):
            console.print("  [green]mlp_extractor:[/green]")
            console.print(policy.mlp_extractor)

        if hasattr(policy, "action_net"):
            console.print("  [green]action_net:[/green]")
            console.print(policy.action_net)

        if hasattr(policy, "value_net"):
            console.print("  [green]value_net:[/green]")
            console.print(policy.value_net)


def _to_effective_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
    """
    Convert the parsed TrainConfig (dataclass schema) back into the input YAML schema.

    Motivation:
    - config_resolved.json is a normalized dataclass dump (machine-friendly).
    - config_effective.yaml should be rerunnable as an input config (human-friendly).
    """
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


def train(*, cfg: TrainConfig, resume_override: Optional[str] = None) -> RunPaths:
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

    # Log effective PPO params + policy network (CNN, heads, etc.)
    log_ppo_params(model, cfg, paths=paths, show_policy_network=True)

    callbacks = make_callbacks(
        cfg=cfg,
        checkpoint_dir=paths.checkpoint_dir,
    )

    print(f"[train] run_id={paths.run_id}")
    print(f"[train] run_dir={paths.run_dir}")
    print(f"[train] tb_dir={paths.tb_dir}")
    print(f"[train] checkpoints={paths.checkpoint_dir}")
    print(f"[train] obs_space={vec_env.observation_space}")
    print(f"[train] action_space={vec_env.action_space}")
    print(f"[train] torch={torch.__version__} cuda={torch.cuda.is_available()}")

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

        print(
            f"[eval-final] mean_reward={metrics['mean_reward']:.6g} "
            f"std_reward={metrics['std_reward']:.6g} "
            f"mean_len={metrics['mean_length']:.3f}"
        )

    vec_env.close()
    print(f"[train] Done. Final model saved to: {final_path}")
    return paths
