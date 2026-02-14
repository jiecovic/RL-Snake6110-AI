# src/snake_rl/rl/train/train_loop.py
from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.rl.callbacks.factory import make_callbacks
from snake_rl.rl.callbacks.progress_bar import SnakeProgressBarCallback
from snake_rl.rl.envs.factory import make_vec_env
from snake_rl.rl.eval.eval_utils import evaluate_model
from snake_rl.rl.metrics import Metrics
from snake_rl.rl.models.model_factory import make_or_load_model
from snake_rl.rl.reporting import log_ppo_params, save_manifest
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.runs.checkpoints import append_jsonl, atomic_save_zip
from snake_rl.utils.runs.run_paths import RunPaths


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
            overwrite=resume_path is None,
        )

        model = make_or_load_model(
            cfg=cfg,
            vec_env=vec_env,
            tensorboard_log=paths.tb_dir,
            resume_path=resume_path,
        )

        logger.info("[train] run:")
        logger.info(f"  id={paths.run_id}")
        logger.info(f"  dir={paths.run_dir}")
        logger.info(f"  tb={paths.tb_dir}")
        logger.info(f"  checkpoints={paths.checkpoint_dir}")
        logger.info(f"  seed={cfg.run.seed} vec={cfg.run.vec} num_envs={cfg.run.num_envs}")
        logger.info(
            f"  total_timesteps={cfg.run.total_timesteps:,} "
            f"checkpoint_freq={cfg.run.checkpoint.freq:,}"
        )
        logger.info(
            f"  eval: enabled={cfg.run.checkpoint.eval.enabled} "
            f"episodes={cfg.run.checkpoint.eval.episodes} "
            f"deterministic={cfg.run.checkpoint.eval.deterministic} "
            f"seed_offset={cfg.run.checkpoint.eval.seed_offset}"
        )
        logger.info("[train] env:")
        logger.info(f"  obs_space={vec_env.observation_space}")
        logger.info(f"  action_space={vec_env.action_space}")
        logger.info(
            f"  obs_spec={cfg.env.obs.kind}/{cfg.env.obs.view} "
            f"frame_stack={cfg.env.frame_stack.n_frames}"
        )
        logger.info(f"  action_spec={cfg.env.action.type}")
        logger.info(f"  board={cfg.board.width}x{cfg.board.height} food={cfg.board.food_count}")
        logger.info(f"[train] torch={torch.__version__} cuda={torch.cuda.is_available()}")

        log_ppo_params(model=model, cfg=cfg, paths=paths, logger=logger, label="train")

        callbacks = make_callbacks(
            cfg=cfg,
            checkpoint_dir=paths.checkpoint_dir,
        )
        if use_rich:
            callbacks = CallbackList([callbacks, SnakeProgressBarCallback()])

        learn_timesteps = int(cfg.run.total_timesteps)
        if resume_path is not None:
            try:
                current_steps = int(getattr(model, "num_timesteps", 0))
            except Exception:
                current_steps = 0
            if current_steps > 0:
                if learn_timesteps <= current_steps:
                    logger.warning(
                        "[train] resume: target total_timesteps (%s) <= current (%s); "
                        "nothing to do.",
                        f"{learn_timesteps:,}",
                        f"{current_steps:,}",
                    )
                    learn_timesteps = 0
                else:
                    remaining = learn_timesteps - current_steps
                    logger.info(
                        "[train] resume: current=%s target=%s remaining=%s",
                        f"{current_steps:,}",
                        f"{learn_timesteps:,}",
                        f"{remaining:,}",
                    )
                    learn_timesteps = remaining

        if resume_path is not None and learn_timesteps <= 0:
            logger.info("[train] resume: no remaining timesteps; skipping learn.")
            finished_ok = True
            return paths

        model.learn(
            total_timesteps=int(learn_timesteps),
            progress_bar=False,
            callback=callbacks,
            tb_log_name="ppo",
            reset_num_timesteps=resume_path is None,
        )

        final_path = paths.checkpoint_dir / "final.zip"
        atomic_save_zip(model=model, dst=final_path)

        if bool(cfg.run.checkpoint.eval.enabled):
            seed_base = int(cfg.run.seed) + int(cfg.run.checkpoint.eval.seed_offset)
            metrics = evaluate_model(
                model=model,
                cfg=cfg,
                episodes=int(cfg.run.checkpoint.eval.episodes),
                deterministic=bool(cfg.run.checkpoint.eval.deterministic),
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
                f"[eval-final] mean_reward={metrics[Metrics.EP_RETURN_MEAN]:.6g} "
                f"std_reward={metrics[Metrics.EP_RETURN_STD]:.6g} "
                f"mean_len={metrics[Metrics.EP_LENGTH_MEAN]:.3f}"
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
