# src/snake_rl/app/train_app.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from snake_rl.config.loader import (
    dataclass_from_raw,
    load_from_hydra_cfg,
    load_train_config_from_path,
)
from snake_rl.config.pydantic_models import TrainConfigModel
from snake_rl.rl.train.train_loop import train
from snake_rl.utils.runs.paths import repo_root, runs_root
from snake_rl.utils.runs.resume import resolve_resume_context
from snake_rl.utils.runs.run_paths import make_run_paths, run_paths_from_dir

CONFIG_DIR = Path(repo_root()) / "configs"


def _is_under_config_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(CONFIG_DIR.resolve())
        return True
    except Exception:
        return False


def _apply_resume_override(
    *,
    cfg: Any,
    model: TrainConfigModel,
    raw_yaml: str | None,
    resume_override: str | None,
    resume_total_steps: int | None,
    resume_add_steps: int | None,
) -> tuple[Any, TrainConfigModel, str | None]:
    if resume_total_steps is not None and resume_add_steps is not None:
        raise ValueError("Use only one of --resume-total-steps or --resume-add-steps.")

    if not resume_override and resume_total_steps is None and resume_add_steps is None:
        return cfg, model, raw_yaml

    raw = model.model_dump(mode="python")
    run = dict(raw.get("run") or {})
    if resume_override:
        run["resume_checkpoint"] = str(resume_override)
    if resume_total_steps is not None:
        if int(resume_total_steps) <= 0:
            raise ValueError("--resume-total-steps must be > 0")
        run["total_timesteps"] = int(resume_total_steps)
    if resume_add_steps is not None:
        extra = int(resume_add_steps)
        if extra <= 0:
            raise ValueError("--resume-add-steps must be > 0")
        base = int(run.get("total_timesteps", 0))
        run["total_timesteps"] = base + extra
    raw["run"] = run

    model = TrainConfigModel.model_validate(raw)
    cfg = model.to_dataclass()
    return cfg, model, raw_yaml


def run_from_config_path(
    *,
    config_path: Path,
    overrides: list[str] | None = None,
    no_rich: bool | None = None,
    log_level: str | None = None,
    resume_override: str | None = None,
    resume_total_steps: int | None = None,
    resume_add_steps: int | None = None,
) -> None:
    cfg_path = Path(config_path)
    cfg, logging_cfg, model, raw_yaml = load_train_config_from_path(
        config_path=cfg_path,
        config_root=CONFIG_DIR,
        overrides=list(overrides or []),
    )
    cfg, model, raw_yaml = _apply_resume_override(
        cfg=cfg,
        model=model,
        raw_yaml=raw_yaml,
        resume_override=resume_override,
        resume_total_steps=resume_total_steps,
        resume_add_steps=resume_add_steps,
    )
    config_hydra_yaml = raw_yaml if _is_under_config_root(cfg_path) else None

    cfg_no_rich = bool(logging_cfg.get("no_rich", False))
    cfg_log_level = str(logging_cfg.get("level", "INFO"))
    use_rich = not cfg_no_rich
    if no_rich is True:
        use_rich = False
    effective_log_level = cfg_log_level if log_level is None else str(log_level)

    resume_path = None
    resume_run_dir = None
    if cfg.run.resume_checkpoint:
        resume_ctx = resolve_resume_context(
            str(cfg.run.resume_checkpoint),
            runs_root=runs_root(),
            legacy_root=repo_root() / "experiments",
        )
        resume_path = resume_ctx.checkpoint_path
        resume_run_dir = resume_ctx.run_dir

    paths = (
        run_paths_from_dir(run_dir=resume_run_dir)
        if resume_run_dir is not None
        else make_run_paths(run_name=str(cfg.run.name))
    )

    train(
        cfg=cfg,
        paths=paths,
        resume_path=resume_path,
        use_rich=use_rich,
        log_level=effective_log_level,
        config_hydra_yaml=config_hydra_yaml,
        config_validated=model.model_dump(mode="python"),
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raw, raw_yaml = load_from_hydra_cfg(cfg)
    train_cfg, logging_cfg, model = dataclass_from_raw(raw)

    no_rich = bool(logging_cfg.get("no_rich", False))
    log_level = str(logging_cfg.get("level", "INFO"))

    resume_path = None
    resume_run_dir = None
    if train_cfg.run.resume_checkpoint:
        resume_ctx = resolve_resume_context(
            str(train_cfg.run.resume_checkpoint),
            runs_root=runs_root(),
            legacy_root=repo_root() / "experiments",
        )
        resume_path = resume_ctx.checkpoint_path
        resume_run_dir = resume_ctx.run_dir

    paths = (
        run_paths_from_dir(run_dir=resume_run_dir)
        if resume_run_dir is not None
        else make_run_paths(run_name=str(train_cfg.run.name))
    )

    train(
        cfg=train_cfg,
        paths=paths,
        resume_path=resume_path,
        use_rich=not no_rich,
        log_level=log_level,
        config_hydra_yaml=raw_yaml,
        config_validated=model.model_dump(mode="python"),
    )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
