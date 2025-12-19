# src/snake_rl/training/train_loop.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.reporting import append_jsonl, log_ppo_params, save_manifest
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths
from snake_rl.utils.checkpoints import atomic_save_zip
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.paths import repo_root


def train(
        *,
        cfg: TrainConfig,
        resume_override: Optional[str] = None,
        use_rich: bool = True,
        log_level: str = "INFO",
) -> RunPaths:
    logger = setup_logger(name="snake_rl.train", use_rich=use_rich, level=log_level)

    # Do as much as possible before creating a run dir.
    set_random_seed(int(cfg.run.seed))

    vec_env = make_vec_env(cfg=cfg)

    paths: Optional[RunPaths] = None
    finished_ok = False

    try:
        # Resolve resume path without needing RunPaths yet.
        experiments_root = repo_root() / "experiments"
        resume_path: Optional[Path] = None
        resume_value = resume_override if resume_override is not None else cfg.run.resume_checkpoint
        if resume_value:
            resume_path = resolve_resume_arg(str(resume_value), experiments_root)

        # Only now create the run directory + snapshot.
        paths = make_run_paths(run_name=str(cfg.run.name))
        save_manifest(run_dir=paths.run_dir, cfg=cfg)

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

        if bool(cfg.eval.final.enabled):
            seed_base = int(cfg.run.seed) + int(cfg.eval.final.seed_offset)
            metrics = evaluate_model(
                model=model,
                cfg=cfg,
                episodes=int(cfg.eval.final.episodes),
                deterministic=bool(cfg.eval.final.deterministic),
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
        if paths is not None:
            try:
                (paths.run_dir / "status.txt").write_text(
                    "finished\n" if finished_ok else "failed\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
