# src/snake_rl/app/train_app.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig

from snake_rl.config.loader import (
    dataclass_from_raw,
    load_from_hydra_cfg,
    load_train_config_from_path,
)
from snake_rl.config.pydantic_models import TrainConfigModel
from snake_rl.rl.train.train_loop import train
from snake_rl.utils.runs.paths import repo_root, resolve_run_dir, runs_root
from snake_rl.utils.runs.resume import resolve_resume_arg
from snake_rl.utils.runs.run_paths import make_run_paths

CONFIG_DIR = Path(repo_root()) / "configs"


def infer_config_path_from_resume(resume: str) -> Path:
    """
    Resolve a resume argument to a run's config_snapshot.yaml.
    Falls back to config_validated.yaml or config_hydra.yaml if snapshot is missing.
    """
    repo = repo_root()
    legacy_root = repo / "experiments"
    p = Path(resume).expanduser()

    run_dir: Path | None = None
    if p.is_dir():
        run_dir = resolve_run_dir(repo, str(p))
    elif p.is_file() and p.suffix.lower() == ".zip":
        run_dir = p.parent.parent
    else:
        try:
            ckpt = resolve_resume_arg(resume, runs_root=runs_root(), legacy_root=legacy_root)
        except Exception:
            run_dir = None
        else:
            run_dir = ckpt.parent.parent

    if run_dir is None:
        raise FileNotFoundError(
            "Could not infer run config from --resume. Pass --config or use a run id/path."
        )

    for name in ("config_snapshot.yaml", "config_validated.yaml", "config_hydra.yaml"):
        cand = run_dir / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No config snapshot found in run dir: {run_dir}")


def _apply_resume_override(
    *,
    cfg: Any,
    model: TrainConfigModel,
    raw_yaml: str | None,
    resume_override: str | None,
) -> tuple[Any, TrainConfigModel, str | None]:
    if not resume_override:
        return cfg, model, raw_yaml

    raw = model.model_dump(mode="python")
    run = dict(raw.get("run") or {})
    run["resume_checkpoint"] = str(resume_override)
    raw["run"] = run

    model = TrainConfigModel.model_validate(raw)
    cfg = model.to_dataclass()
    raw_yaml = yaml.safe_dump(raw, sort_keys=False)
    return cfg, model, raw_yaml


def run_from_config_path(
    *,
    config_path: Path,
    overrides: list[str] | None = None,
    no_rich: bool | None = None,
    log_level: str | None = None,
    resume_override: str | None = None,
) -> None:
    cfg_path = Path(config_path)
    cfg, logging_cfg, model, raw_yaml = load_train_config_from_path(
        config_path=cfg_path,
        config_root=CONFIG_DIR,
        overrides=list(overrides or []),
    )
    cfg, model, raw_yaml = _apply_resume_override(
        cfg=cfg, model=model, raw_yaml=raw_yaml, resume_override=resume_override
    )

    cfg_no_rich = bool(logging_cfg.get("no_rich", False))
    cfg_log_level = str(logging_cfg.get("level", "INFO"))
    use_rich = not cfg_no_rich
    if no_rich is True:
        use_rich = False
    effective_log_level = cfg_log_level if log_level is None else str(log_level)

    resume_path = None
    if cfg.run.resume_checkpoint:
        resume_path = resolve_resume_arg(
            str(cfg.run.resume_checkpoint),
            runs_root=runs_root(),
            legacy_root=repo_root() / "experiments",
        )

    paths = make_run_paths(run_name=str(cfg.run.name))

    train(
        cfg=cfg,
        paths=paths,
        resume_path=resume_path,
        use_rich=use_rich,
        log_level=effective_log_level,
        config_hydra_yaml=raw_yaml,
        config_validated=model.model_dump(mode="python"),
    )


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raw, raw_yaml = load_from_hydra_cfg(cfg)
    train_cfg, logging_cfg, model = dataclass_from_raw(raw)

    no_rich = bool(logging_cfg.get("no_rich", False))
    log_level = str(logging_cfg.get("level", "INFO"))

    resume_path = None
    if train_cfg.run.resume_checkpoint:
        resume_path = resolve_resume_arg(
            str(train_cfg.run.resume_checkpoint),
            runs_root=runs_root(),
            legacy_root=repo_root() / "experiments",
        )

    paths = make_run_paths(run_name=str(train_cfg.run.name))

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
