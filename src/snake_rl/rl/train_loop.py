# src/snake_rl/rl/train_loop.py
from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.rl.callbacks_factory import make_callbacks
from snake_rl.rl.env_factory import make_vec_env
from snake_rl.rl.eval_utils import evaluate_model
from snake_rl.rl.model_factory import make_or_load_model
from snake_rl.rl.reporting import log_ppo_params, save_manifest
from snake_rl.utils.checkpoints import append_jsonl, atomic_save_zip
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.run_paths import RunPaths


def train(
    *,
    cfg: TrainConfig,
    paths: RunPaths,
    resume_path: Path | None = None,
    use_rich: bool = True,
    log_level: str = "INFO",
    config_hydra_yaml: str | None = None,
    config_validated: dict[str, Any] | None = None,
) -> RunPaths:
    logger = setup_logger(name="snake_rl.train", use_rich=use_rich, level=log_level)

    # Do as much as possible before creating a run dir.
    set_random_seed(int(cfg.run.seed))

    vec_env = make_vec_env(cfg=cfg)

    finished_ok = False

    try:
        save_manifest(
            run_dir=paths.run_dir,
            cfg=cfg,
            hydra_yaml=config_hydra_yaml,
            validated_cfg=config_validated,
        )

        model = make_or_load_model(
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

        callbacks = make_callbacks(cfg=cfg, checkpoint_dir=paths.checkpoint_dir)

        model.learn(
            total_timesteps=int(cfg.run.total_timesteps),
            progress_bar=bool(use_rich),
            callback=callbacks,
            tb_log_name="ppo",
        )

        final_path = paths.checkpoint_dir / "final.zip"
        atomic_save_zip(model=model, dst=final_path)

        if bool(cfg.train.eval.enabled):
            seed_base = int(cfg.run.seed) + int(cfg.train.eval.seed_offset)
            metrics = evaluate_model(
                model=model,
                cfg=cfg,
                episodes=int(cfg.train.eval.episodes),
                deterministic=bool(cfg.train.eval.deterministic),
                seed_base=seed_base,
                num_envs=1,
            )
            metrics["phase"] = "final"
            metrics["timesteps"] = int(cfg.run.total_timesteps)

            (paths.run_dir / "eval_final.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            append_jsonl(paths.checkpoint_dir / "eval_history.jsonl", metrics)

            logger.info(
                f"[eval-final] mean_reward={metrics['mean_reward']:.6g} "
                f"std_reward={metrics['std_reward']:.6g} "
                f"mean_len={metrics['mean_length']:.3f}"
            )

        logger.info(f"[train] Done. Final model saved to: {final_path}")
        finished_ok = True
        return paths

    finally:
        vec_env.close()
        # If anything fails after run dir creation, leave a minimal marker.
        with suppress(Exception):
            (paths.run_dir / "status.txt").write_text(
                "finished\n" if finished_ok else "failed\n",
                encoding="utf-8",
            )
