# src/snake_rl/app/watch_app.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from snake_rl import _core as core
from snake_rl.config.access import (
    get_board_params,
    get_env_action,
    get_env_obs,
    get_frame_stack_n,
)
from snake_rl.config.loader import load_train_config_from_path
from snake_rl.envs.snake_env import SnakeEnv
from snake_rl.envs.specs import ActionSpec, ObservationSpec
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snake_engine import SnakeEngine
from snake_rl.rl.reporting import log_ppo_params
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.model_params import format_sb3_param_summary
from snake_rl.utils.models import load_ppo
from snake_rl.utils.obs import sanitize_observation
from snake_rl.utils.runs.checkpoints import pick_checkpoint
from snake_rl.utils.runs.paths import relpath, repo_root, resolve_run_dir


def _make_engine_from_board_params(board: dict[str, int], frame_stack_n: int) -> SnakeEngine:
    core_board = core.Board(width=int(board["width"]), height=int(board["height"]))
    return SnakeEngine(
        board=core_board,
        food_count=int(board["food_count"]),
        frame_stack_n=int(frame_stack_n),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake (pygame).")
    p.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run id / run folder name under runs/ (legacy: experiments/)",
    )
    p.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["auto", "latest", "best", "best_reward", "best_score", "best_win", "final"],
    )
    p.add_argument(
        "--reload",
        type=float,
        default=5.0,
        help="Poll for newer checkpoint every N seconds (0 = disable).",
    )
    p.add_argument("--fps", type=int, default=0, help="Render FPS cap (0 = uncapped).")
    p.add_argument("--sim-hz", type=int, default=25, help="Simulation steps per second.")
    p.add_argument("--pixel-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich logging (fallback to plain logging).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config path. If omitted, uses runs/<run>/config_snapshot.yaml. "
            "If path is under configs/, Hydra defaults + overrides are applied."
        ),
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable). Only used when --config is under configs/.",
    )

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
        self.last_reload_at = time.time()
        self.win_count = 0
        self.episode_return = 0.0
        self.last_reward = 0.0

        self.logger = logger
        self.repo = repo

        self.obs = self.vec_env.reset()

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
                self.last_reload_at = time.time()
                rel = relpath(chosen, base=self.repo)
                self.logger.info(f"reloaded checkpoint: {rel} (mtime={int(mtime)})")
        except Exception:
            self.logger.exception("reload error")

    def reload_age_seconds(self) -> float:
        return max(0.0, time.time() - self.last_reload_at)

    def step(self, action_override: int | None = None) -> None:
        self.maybe_reload()

        if action_override is None:
            obs_for_model = sanitize_observation(self.obs)
            action, _ = self.model.predict(obs_for_model, deterministic=True)

            if np.isscalar(action):
                if isinstance(action, (bool, int, np.integer, np.floating)):
                    act = np.array([int(action)], dtype=np.int64)
                else:
                    raise TypeError(f"Unsupported scalar action type: {type(action)!r}")
            else:
                act = np.asarray(action, dtype=np.int64).reshape((1,))
        else:
            act = np.asarray([int(action_override)], dtype=np.int64).reshape((1,))

        obs_next, reward, dones, infos = self.vec_env.step(act)
        done0 = bool(np.asarray(dones).reshape((-1,))[0])
        reward0 = float(np.asarray(reward).reshape((-1,))[0])
        self.last_reward = reward0
        self.episode_return += reward0

        info0 = None
        if isinstance(infos, (list, tuple)) and infos:
            info0 = infos[0]
        elif isinstance(infos, dict):
            info0 = infos
        if isinstance(info0, dict):
            cause = str(info0.get("termination_cause", "")).strip().lower()
            if cause == "win":
                self.win_count += 1
            else:
                move_results = info0.get("move_results")
                if move_results is not None:
                    try:
                        if int(move_results) & int(core.MOVE_WIN):
                            self.win_count += 1
                    except Exception:
                        pass

        if done0:
            self.obs = self.vec_env.reset()
            self.episode_return = 0.0
        else:
            self.obs = obs_next


def main() -> None:
    args = parse_args()

    logger = setup_logger(
        name="snake_rl.watch",
        use_rich=not bool(args.no_rich),
        level=str(args.log_level),
    )

    repo = repo_root()
    run_dir = resolve_run_dir(repo, args.run)

    config_path = Path(args.config) if args.config else (run_dir / "config_snapshot.yaml")
    cfg, _, _, _ = load_train_config_from_path(
        config_path=config_path,
        config_root=repo / "configs",
        overrides=list(args.override),
    )

    ckpt = None
    poll_s = float(args.reload) if args.reload and float(args.reload) > 0 else 1.0
    warned_wait = False
    while ckpt is None:
        try:
            ckpt = pick_checkpoint(run_dir=run_dir, which=args.which)
        except FileNotFoundError:
            if not warned_wait:
                logger.info("waiting for first checkpoint...")
                warned_wait = True
            time.sleep(poll_s)
            continue

    model = load_ppo(ckpt, device=str(args.device))
    logger.info(f"loaded checkpoint: {relpath(ckpt, base=repo)}")
    logger.info(format_sb3_param_summary(model))
    log_ppo_params(
        model=model,
        cfg=cfg,
        paths=SimpleNamespace(repo_root=repo),
        logger=logger,
        label="watch",
    )

    n_stack = int(get_frame_stack_n(cfg))
    board = get_board_params(cfg)
    game = _make_engine_from_board_params(board, frame_stack_n=int(n_stack))

    obs_cfg = dict(get_env_obs(cfg))
    obs_spec = ObservationSpec(
        kind=str(obs_cfg.get("kind")),
        view=str(obs_cfg.get("view")),
        params=dict(obs_cfg.get("params", {})),
        features=dict(obs_cfg.get("features", {})),
    )
    obs_spec.kind_norm()
    obs_spec.view_norm()
    action_spec = ActionSpec(type=str(get_env_action(cfg)))

    base_env = SnakeEnv(
        game,
        obs=obs_spec,
        action=action_spec,
        frame_stack_n=int(n_stack),
    )

    # IMPORTANT:
    # Seeding happens at the ENV level, not the game.
    # This will create env.np_random and inject it into SnakeEngine via reset().
    base_env.reset(seed=int(args.seed))

    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecMonitor(vec_env)

    controller = WatchController(
        vec_env=vec_env,
        model=model,
        run_dir=run_dir,
        which=str(args.which),
        device=str(args.device),
        reload_seconds=float(args.reload),
        initial_ckpt=ckpt,
        logger=logger,
        repo=repo,
    )

    hud_info = {
        "mode": "watch",
        "run": run_dir.name,
        "which": str(args.which),
        "seed": str(args.seed),
        "obs": f"{obs_spec.kind_norm()}/{obs_spec.view_norm()}",
        "action": str(action_spec.type),
    }

    def _controller_step(action_override: int | None = None) -> None:
        controller.step(action_override)
        hud_info["reload"] = f"{controller.reload_age_seconds():.1f}s"
        hud_info["wins"] = str(controller.win_count)
        hud_info["reward"] = f"{controller.last_reward:+.3f}"
        hud_info["ep_return"] = f"{controller.episode_return:.3f}"

    try:
        run_pygame_app(
            game=game,
            cfg=AppConfig(
                fps=int(args.fps),
                sim_hz=int(args.sim_hz),
                pixel_size=int(args.pixel_size),
                caption=f"Snake (watch: {run_dir.name} / {args.which})",
                enable_human_input=True,
                agent_view_spec=obs_spec,
                agent_view_vocab_name=obs_spec.tile_vocab_name(),
                agent_view_vocab_num_classes=obs_spec.tile_vocab_num_classes(),
                agent_view_side="left",
                hud_mode="selected",
                hud_features=obs_spec.features,
                hud_info=hud_info,
            ),
            step_fn=_controller_step,
        )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
