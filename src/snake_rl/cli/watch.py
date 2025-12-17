# src/snake_rl/cli/watch.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake (pygame).")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--which", type=str, default="best", choices=["auto", "latest", "best", "final"])
    p.add_argument("--reload", type=float, default=0.0, help="If >0, poll for newer checkpoint every N seconds.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--pixel-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
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
                self.model = _load_model(chosen, device=self.device)
                self.current_ckpt = chosen
                self.current_mtime = mtime
                print(f"[watch] reloaded checkpoint: {chosen}", flush=True)
        except Exception as e:
            print("[watch] reload error:", repr(e), flush=True)

    def step(self) -> None:
        self.maybe_reload()

        obs_for_model = sanitize_observation(self.obs)
        action, _ = self.model.predict(obs_for_model, deterministic=True)

        if np.isscalar(action):
            act = np.array([int(action)], dtype=np.int64)
        else:
            act = np.asarray(action, dtype=np.int64).reshape((1,))

        obs_next, _reward, dones, _infos = self.vec_env.step(act)
        done0 = bool(np.asarray(dones).reshape((-1,))[0])

        self.obs = self.vec_env.reset() if done0 else obs_next


def main() -> None:
    args = parse_args()

    repo = repo_root()
    run_dir = resolve_run_dir(repo, args.run)
    cfg_d = load_config_resolved(run_dir=run_dir)

    ckpt = pick_checkpoint(run_dir=run_dir, which=args.which)
    model = _load_model(ckpt, device=str(args.device))
    print(f"[watch] loaded checkpoint: {ckpt}", flush=True)

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
            vec_env = DictPixelVecFrameStack(vec_env, n_stack=n_stack, pixel_key="pixel")
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
