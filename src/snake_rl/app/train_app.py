# src\snake_rl\app\train_app.py
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from snake_rl.config.loader import dataclass_from_raw, load_from_hydra_cfg
from snake_rl.rl.train_loop import train
from snake_rl.utils.paths import repo_root, runs_root
from snake_rl.utils.resume import resolve_resume_arg
from snake_rl.utils.run_paths import make_run_paths

CONFIG_DIR = Path(repo_root()) / "configs"


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
