# src/snake_rl/cli/watch.py
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from snake_rl.config.snapshot import (
    get_env_id,
    get_env_params,
    get_frame_stack_n,
    get_level_params,
    load_snapshot_config,
)
from snake_rl.envs.registry import get_env_cls
from snake_rl.game.level import EmptyLevel
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snakegame import SnakeGame
from snake_rl.tools.obs_debug import debug_print_obs
from snake_rl.training.env_factory import apply_frame_stack
from snake_rl.utils.checkpoints import pick_checkpoint
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.model_params import format_sb3_param_report, format_sb3_param_summary
from snake_rl.utils.models import load_ppo
from snake_rl.utils.obs import sanitize_observation
from snake_rl.utils.paths import relpath, repo_root, resolve_run_dir


def _ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _make_game_from_level_params(level: dict[str, int], *, seed: int) -> SnakeGame:
    lvl = EmptyLevel(height=int(level["height"]), width=int(level["width"]))
    return SnakeGame(level=lvl, food_count=int(level["food_count"]), seed=int(seed))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake (pygame).")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--which", type=str, default="best", choices=["auto", "latest", "best", "final"])
    p.add_argument("--reload", type=float, default=0.0, help="If >0, poll for newer checkpoint every N seconds.")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--pixel-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging (fallback to plain logging).")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    p.add_argument("--print-obs", action="store_true")
    p.add_argument("--print-obs-every", type=int, default=1)
    p.add_argument("--print-obs-max", type=int, default=5)
    return p.parse_args()


class WatchController:
    def __init__(
            self,
            *,
            vec_env,
            model: PPO,
            run_dir: Path,
            which: str,
            device: str,
            reload_seconds: float,
            initial_ckpt: Path,
            print_obs: bool,
            print_obs_every: int,
            print_obs_max: int,
            logger,
            repo: Path,
    ):
        self.vec_env = vec_env
        self.model = model

        self.run_dir = run_dir
        self.which = which
        self.device = device

        self.reload_seconds = float(reload_seconds)
        self.current_ckpt = initial_ckpt
        self.current_mtime = initial_ckpt.stat().st_mtime
        self.last_reload_check = 0.0

        self.print_obs = bool(print_obs)
        self.print_obs_every = max(1, int(print_obs_every))
        self.print_obs_max = max(0, int(print_obs_max))
        self._step_i = 0
        self._printed = 0

        self.logger = logger
        self.repo = repo

        self.obs = self.vec_env.reset()

        if self.print_obs and (self.print_obs_max == 0 or self._printed < self.print_obs_max):
            self.logger.info(f"obs dump after reset() [{_ts()}]")
            debug_print_obs(self.obs, header="[watch][obs] raw observation:")
            self._printed += 1

    def maybe_reload(self) -> None:
        if not (self.reload_seconds and self.reload_seconds > 0):
            return

        now = time.time()
        if (now - self.last_reload_check) < self.reload_seconds:
            return
        self.last_reload_check = now

        try:
            chosen = pick_checkpoint(run_dir=self.run_dir, which=self.which)
            mtime = chosen.stat().st_mtime
            if chosen != self.current_ckpt or mtime > self.current_mtime:
                self.model = load_ppo(chosen, device=self.device)
                self.current_ckpt = chosen
                self.current_mtime = mtime
                self.logger.info(
                    f"reloaded checkpoint: {relpath(chosen, base=self.repo)} "
                    f"(mtime={int(mtime)})"
                )
        except Exception:
            self.logger.exception("reload error")

    def step(self) -> None:
        self.maybe_reload()

        if self.print_obs:
            do_print = (self._step_i % self.print_obs_every) == 0
            under_limit = (self.print_obs_max == 0) or (self._printed < self.print_obs_max)
            if do_print and under_limit:
                self.logger.info(f"obs dump step={self._step_i} [{_ts()}]")
                debug_print_obs(self.obs, header="[watch][obs] raw observation:")
                self._printed += 1

        obs_for_model = sanitize_observation(self.obs)
        action, _ = self.model.predict(obs_for_model, deterministic=True)

        if np.isscalar(action):
            act = np.array([int(action)], dtype=np.int64)
        else:
            act = np.asarray(action, dtype=np.int64).reshape((1,))

        obs_next, _reward, dones, _infos = self.vec_env.step(act)
        done0 = bool(np.asarray(dones).reshape((-1,))[0])

        self.obs = self.vec_env.reset() if done0 else obs_next
        self._step_i += 1


def main() -> None:
    args = parse_args()

    logger = setup_logger(
        name="snake_rl.watch",
        use_rich=not bool(args.no_rich),
        level=str(args.log_level),
    )

    repo = repo_root()
    run_dir = resolve_run_dir(repo, args.run)

    cfg = load_snapshot_config(run_dir=run_dir)

    ckpt = pick_checkpoint(run_dir=run_dir, which=args.which)
    model = load_ppo(ckpt, device=str(args.device))
    logger.info(f"loaded checkpoint: {relpath(ckpt, base=repo)}")
    logger.info(format_sb3_param_summary(model))
    logger.info(format_sb3_param_report(model))

    level = get_level_params(cfg)
    game = _make_game_from_level_params(level, seed=int(args.seed))

    env_id = get_env_id(cfg)
    env_cls = get_env_cls(env_id)
    env_params = get_env_params(cfg)

    base_env = env_cls(game, **env_params)  # type: ignore[arg-type]

    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecMonitor(vec_env)

    n_stack = int(get_frame_stack_n(cfg))
    vec_env = apply_frame_stack(vec_env=vec_env, n_stack=n_stack, pixel_key="pixel")

    controller = WatchController(
        vec_env=vec_env,
        model=model,
        run_dir=run_dir,
        which=str(args.which),
        device=str(args.device),
        reload_seconds=float(args.reload),
        initial_ckpt=ckpt,
        print_obs=bool(args.print_obs),
        print_obs_every=int(args.print_obs_every),
        print_obs_max=int(args.print_obs_max),
        logger=logger,
        repo=repo,
    )

    run_pygame_app(
        game=game,
        cfg=AppConfig(
            fps=int(args.fps),
            pixel_size=int(args.pixel_size),
            caption=f"Snake (watch: {run_dir.name} / {args.which})",
            enable_human_input=False,
        ),
        step_fn=controller.step,
    )

    vec_env.close()


if __name__ == "__main__":
    main()
