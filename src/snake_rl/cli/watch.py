# src/snake_rl/cli/watch.py
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from snake_rl.envs.registry import ENV_REGISTRY
from snake_rl.game.level import EmptyLevel
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snakegame import SnakeGame
from snake_rl.training.eval_utils import sanitize_observation
from snake_rl.training.env_factory import DictPixelVecFrameStack
from snake_rl.utils.checkpoints import pick_checkpoint
from snake_rl.utils.paths import repo_root, resolve_run_dir
from snake_rl.utils.run_config import (
    get_env_id,
    get_env_params,
    get_frame_stack_n,
    get_level_params,
    load_config_resolved,
)


def _ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _relpath(p: Path, *, base: Path) -> str:
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(p)


def _setup_logging(*, use_rich: bool, level: str) -> logging.Logger:
    logger = logging.getLogger("snake_rl.watch")
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


def _load_model(model_path: Path, *, device: str) -> PPO:
    return PPO.load(str(model_path), device=device)


def _make_game_from_level_params(level: dict[str, int], *, seed: int) -> SnakeGame:
    lvl = EmptyLevel(height=int(level["height"]), width=int(level["width"]))
    return SnakeGame(level=lvl, food_count=int(level["food_count"]), seed=int(seed))


def _get_env_cls(env_id: str):
    if env_id not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}")
    return ENV_REGISTRY[env_id]


def _np_summary(x: np.ndarray) -> str:
    try:
        mn = x.min()
        mx = x.max()
    except Exception:
        mn, mx = "?", "?"
    return f"shape={tuple(x.shape)} dtype={x.dtype} min={mn} max={mx}"


def _print_tile_grid(grid_hw: np.ndarray, *, max_h: int = 13, max_w: int = 22) -> None:
    """
    Pretty-print a 2D uint tile-id grid as integers (clipped to max_h/max_w).
    """
    h, w = grid_hw.shape
    hh = min(h, max_h)
    ww = min(w, max_w)
    sub = grid_hw[:hh, :ww]

    width = max(2, len(str(int(sub.max()))) if sub.size else 2)
    fmt = f"{{:>{width}d}}"

    for y in range(hh):
        row = " ".join(fmt.format(int(v)) for v in sub[y])
        print(row, flush=True)

    if hh < h or ww < w:
        print(f"[watch][obs] (clipped print to {hh}x{ww} of full {h}x{w})", flush=True)


def _debug_print_obs(obs: Any) -> None:
    """
    Print raw VecEnv observation in a robust way.

    Handles:
      - Box obs: np.ndarray with shape (n_envs, C, H, W) or (n_envs, H, W) etc.
      - Dict obs: dict of np.ndarray values

    For tile-id envs (uint ids), you typically get (1, 1, 13, 22) for global_tile_id.
    """
    print("[watch][obs] raw observation:", flush=True)

    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                print(f"  key={k!r}: {_np_summary(v)}", flush=True)
            else:
                print(f"  key={k!r}: type={type(v).__name__}", flush=True)

        for tiles_key in ("tiles", "tile", "grid"):
            if tiles_key in obs and isinstance(obs[tiles_key], np.ndarray):
                arr = obs[tiles_key]
                g = arr[0]
                if g.ndim == 3:
                    g2 = g[0]
                elif g.ndim == 2:
                    g2 = g
                else:
                    g2 = None
                if isinstance(g2, np.ndarray) and g2.ndim == 2:
                    print(f"[watch][obs] printing {tiles_key!r} grid for env0:", flush=True)
                    _print_tile_grid(g2)
                break
        return

    if isinstance(obs, np.ndarray):
        print(f"  {_np_summary(obs)}", flush=True)

        if obs.ndim == 4 and obs.shape[0] >= 1 and obs.shape[2] <= 64 and obs.shape[3] <= 64:
            env0 = obs[0]
            if env0.ndim == 3:
                c, _h, _w = env0.shape
                if c >= 1:
                    grid = env0[0]
                    if grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer):
                        print("[watch][obs] printing env0 channel0 grid:", flush=True)
                        _print_tile_grid(grid)
        return

    print(f"  type={type(obs).__name__}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake (pygame).")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--which", type=str, default="best", choices=["auto", "latest", "best", "final"])
    p.add_argument("--reload", type=float, default=0.0, help="If >0, poll for newer checkpoint every N seconds.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--pixel-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    # Logging cosmetics
    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging (fallback to plain logging).")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    # Observation debug printing
    p.add_argument(
        "--print-obs",
        action="store_true",
        help="Print raw VecEnv observations (shape/dtype/min/max). Useful to verify tile-id grids.",
    )
    p.add_argument(
        "--print-obs-every",
        type=int,
        default=1,
        help="If --print-obs is set, print every N steps (default: 1).",
    )
    p.add_argument(
        "--print-obs-max",
        type=int,
        default=5,
        help="If --print-obs is set, stop after this many prints (default: 5).",
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
            print_obs: bool,
            print_obs_every: int,
            print_obs_max: int,
            logger: logging.Logger,
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
            _debug_print_obs(self.obs)
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
                self.model = _load_model(chosen, device=self.device)
                self.current_ckpt = chosen
                self.current_mtime = mtime
                self.logger.info(
                    f"reloaded checkpoint: {_relpath(chosen, base=self.repo)} "
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
                _debug_print_obs(self.obs)
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

    logger = _setup_logging(use_rich=not bool(args.no_rich), level=str(args.log_level))

    repo = repo_root()
    run_dir = resolve_run_dir(repo, args.run)
    cfg_d = load_config_resolved(run_dir=run_dir)

    ckpt = pick_checkpoint(run_dir=run_dir, which=args.which)
    model = _load_model(ckpt, device=str(args.device))
    logger.info(f"loaded checkpoint: {_relpath(ckpt, base=repo)}")

    level = get_level_params(cfg_d)
    game = _make_game_from_level_params(level, seed=int(args.seed))

    env_id = get_env_id(cfg_d)
    env_cls = _get_env_cls(env_id)
    env_params = get_env_params(cfg_d)

    base_env = env_cls(game, **env_params)  # type: ignore[arg-type]

    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecMonitor(vec_env)

    n_stack = get_frame_stack_n(cfg_d)
    if n_stack > 1:
        obs_space = vec_env.observation_space
        if isinstance(obs_space, spaces.Box):
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        elif isinstance(obs_space, spaces.Dict):
            if "pixel" in obs_space.spaces:
                vec_env = DictPixelVecFrameStack(vec_env, n_stack=n_stack, pixel_key="pixel")
            else:
                raise TypeError("Dict observation has no 'pixel' key; stacking not implemented for this env.")
        else:
            raise TypeError(f"Unsupported observation_space type for stacking: {type(obs_space)}")

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
