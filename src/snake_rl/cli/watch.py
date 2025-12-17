# src/snake_rl/cli/watch.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

from snake_rl.envs.registry import ENV_REGISTRY
from snake_rl.game.level import EmptyLevel
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snakegame import SnakeGame
from snake_rl.training.eval_utils import sanitize_observation

from snake_rl.training.env_factory import DictPixelVecFrameStack  # type: ignore

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file() or (p / ".git").exists():
            return p
    return start.resolve()


def _resolve_run_dir(repo_root: Path, run: str) -> Path:
    p = Path(run)
    if p.exists():
        if p.is_dir():
            if (p / "checkpoints").is_dir():
                return p
            if p.name == "checkpoints":
                return p.parent
        raise FileNotFoundError(f"--run points to an existing path but not a run dir: {p}")
    return repo_root / "experiments" / run


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _pick_checkpoint(*, run_dir: Path, which: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"

    if which in ("latest", "best", "final"):
        p = ckpt_dir / f"{which}.zip"
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    if which == "auto":
        state = _read_json(ckpt_dir / "state.json") or {}
        for key in ("best", "latest"):
            entry = state.get(key, {})
            path = entry.get("path")
            if isinstance(path, str):
                p = run_dir / path
                if p.is_file():
                    return p
        p = ckpt_dir / "latest.zip"
        if p.is_file():
            return p
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    raise ValueError("--which must be one of: auto, latest, best, final")


def _load_model(model_path: Path, *, device: str) -> PPO:
    return PPO.load(str(model_path), device=device)


def _load_run_cfg_minimal(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config_resolved.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Could not find config_resolved.json in run dir: {run_dir}")

    d = cast(dict[str, Any], json.loads(cfg_path.read_text(encoding="utf-8")))

    # minimal sanity
    _ = d["env"]["id"]
    _ = d["env"].get("params", {})
    _ = d["level"]["height"]
    _ = d["level"]["width"]
    _ = d["level"]["food_count"]

    return d


def _make_game_from_cfg_d(cfg_d: dict[str, Any], *, seed: int) -> SnakeGame:
    level_d = cast(dict[str, Any], cfg_d["level"])
    level = EmptyLevel(height=int(level_d["height"]), width=int(level_d["width"]))
    return SnakeGame(level=level, food_count=int(level_d["food_count"]), seed=int(seed))


def _get_env_cls_from_cfg_d(cfg_d: dict[str, Any]):
    env_id = str(cast(dict[str, Any], cfg_d["env"])["id"])
    if env_id not in ENV_REGISTRY:
        available = ", ".join(sorted(ENV_REGISTRY.keys()))
        raise ValueError(f"Unknown env.id={env_id!r}. Available: {available}")
    return ENV_REGISTRY[env_id]


def _get_n_stack_from_cfg_d(cfg_d: dict[str, Any]) -> int:
    fs = cast(dict[str, Any], cfg_d.get("observation", {})).get("frame_stack", {})
    try:
        n = int(cast(dict[str, Any], fs).get("n_frames", 1))
    except Exception:
        n = 1
    return max(1, n)


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


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root(Path.cwd())
    run_dir = _resolve_run_dir(repo_root, args.run)

    cfg_d = _load_run_cfg_minimal(run_dir)

    # --- load model (and allow hot reload) ---
    current_ckpt = _pick_checkpoint(run_dir=run_dir, which=args.which)
    current_mtime = current_ckpt.stat().st_mtime
    model = _load_model(current_ckpt, device=str(args.device))
    print(f"[watch] loaded checkpoint: {current_ckpt}", flush=True)

    # --- build the EXACT same env object (game instance is what pygame renders) ---
    game = _make_game_from_cfg_d(cfg_d, seed=int(args.seed))
    env_cls = _get_env_cls_from_cfg_d(cfg_d)
    env_params = dict(cast(dict[str, Any], cfg_d["env"]).get("params", {}))

    base_env = env_cls(game, **env_params)

    # --- make it a 1-env VecEnv so we can reuse the SAME wrappers as training ---
    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecMonitor(vec_env)

    n_stack = _get_n_stack_from_cfg_d(cfg_d)
    if n_stack > 1:
        obs_space = vec_env.observation_space
        if isinstance(obs_space, spaces.Box):
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        elif isinstance(obs_space, spaces.Dict):
            vec_env = DictPixelVecFrameStack(vec_env, n_stack=n_stack, pixel_key="pixel")
        else:
            raise TypeError(f"Unsupported observation_space type for stacking: {type(obs_space)}")

    # Initial reset gives us the correct stacked vec observation (shape has leading batch dim = 1)
    obs = vec_env.reset()

    reload_state = {"last_check": 0.0}

    def step_fn() -> None:
        nonlocal model, current_ckpt, current_mtime, obs

        # Non-blocking checkpoint polling (never sleep inside pygame loop)
        if args.reload and args.reload > 0:
            now = time.time()
            if (now - reload_state["last_check"]) >= float(args.reload):
                reload_state["last_check"] = now
                try:
                    chosen = _pick_checkpoint(run_dir=run_dir, which=args.which)
                    mtime = chosen.stat().st_mtime
                    if chosen != current_ckpt or mtime > current_mtime:
                        model = _load_model(chosen, device=str(args.device))
                        current_ckpt = chosen
                        current_mtime = mtime
                        print(f"[watch] reloaded checkpoint: {chosen}", flush=True)
                except Exception as e:
                    print("[watch] reload error:", repr(e), flush=True)

        # VecObs -> model
        obs_for_model = sanitize_observation(obs)
        action, _ = model.predict(obs_for_model, deterministic=True)

        # VecEnv step expects batched action for n_envs=1
        if np.isscalar(action):
            act = np.array([int(action)], dtype=np.int64)
        else:
            act = np.asarray(action, dtype=np.int64).reshape((1,))

        obs_next, _reward, dones, _infos = vec_env.step(act)

        # Gymnasium VecEnv returns dones as array-like of shape (n_envs,)
        done0 = bool(np.asarray(dones).reshape((-1,))[0])

        if done0:
            obs = vec_env.reset()
        else:
            obs = obs_next

    run_pygame_app(
        game=game,
        cfg=AppConfig(
            fps=int(args.fps),
            pixel_size=int(args.pixel_size),
            caption=f"Snake (watch: {run_dir.name} / {args.which})",
            enable_human_input=False,
        ),
        step_fn=step_fn,
    )

    vec_env.close()


if __name__ == "__main__":
    main()
