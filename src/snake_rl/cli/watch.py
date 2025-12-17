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

from snake_rl.envs.registry import ENV_REGISTRY
from snake_rl.game.level import EmptyLevel
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snakegame import SnakeGame
from snake_rl.training.eval_utils import sanitize_observation

# IMPORTANT: Watch must drive the pygame loop with the SAME action semantics as training:
# 0=forward, 1=left, 2=right (RelativeDirection).
try:
    from snake_rl.game.geometry import RelativeDirection
except Exception:  # pragma: no cover
    RelativeDirection = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


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


def _get_current_obs_from_env(env: Any) -> Any:
    """
    pygame loop steps SnakeGame directly; we only read env's current obs.
    """
    if hasattr(env, "get_obs"):
        return env.get_obs()
    if hasattr(env, "_get_obs"):
        return env._get_obs()  # noqa: SLF001
    if hasattr(env, "observe"):
        return env.observe()
    raise AttributeError(
        "Env does not expose an observation getter. Expected one of: get_obs(), _get_obs(), observe()."
    )


# -----------------------------------------------------------------------------#
# Frame stacking wrappers for WATCH (single-env, non-VecEnv)
# -----------------------------------------------------------------------------#


class BoxFrameStackWrapper:
    """
    Single-env wrapper: stack Box obs shaped (C,H,W) along channel axis -> (C*n_stack,H,W).
    """

    def __init__(self, env: Any, *, n_stack: int):
        self.env = env
        self.n_stack = int(n_stack)

        obs_space = getattr(env, "observation_space", None)
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("BoxFrameStackWrapper requires Box observation_space")
        if obs_space.shape is None or len(obs_space.shape) != 3:
            raise TypeError(f"Expected Box shape (C,H,W), got {obs_space.shape}")

        c, h, w = map(int, obs_space.shape)
        self._c = c
        self._h = h
        self._w = w

        def _stack_bounds(x):
            if np.isscalar(x):
                return x
            arr = np.asarray(x)  # (C,H,W)
            return np.tile(arr, (self.n_stack, 1, 1))  # (C*n_stack,H,W)

        self.observation_space = spaces.Box(
            low=_stack_bounds(obs_space.low),
            high=_stack_bounds(obs_space.high),
            shape=(c * self.n_stack, h, w),
            dtype=obs_space.dtype,
        )
        self.action_space = getattr(env, "action_space", None)

        self._buf: Optional[np.ndarray] = None  # (C*n_stack,H,W)

    def reset(self, *args, **kwargs):
        if hasattr(self.env, "reset"):
            out = self.env.reset(*args, **kwargs)
        else:
            out = None
        obs = _get_current_obs_from_env(self.env)
        _ = self._reset_buf(obs)
        return out if out is not None else obs

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def _reset_buf(self, obs: Any):
        pix = np.asarray(obs)
        if pix.shape != (self._c, self._h, self._w):
            raise ValueError(f"Expected obs shape {(self._c, self._h, self._w)}, got {pix.shape}")

        stacked = np.concatenate([pix] * self.n_stack, axis=0)
        if self._buf is None or self._buf.shape != stacked.shape or self._buf.dtype != stacked.dtype:
            self._buf = np.zeros_like(stacked)
        self._buf[...] = stacked
        return self._buf.copy()

    def _update_buf(self, obs: Any):
        pix = np.asarray(obs)
        if self._buf is None:
            return self._reset_buf(pix)

        c = self._c
        self._buf[:-c, :, :] = self._buf[c:, :, :]
        self._buf[-c:, :, :] = pix
        return self._buf.copy()

    def get_obs(self):
        obs = _get_current_obs_from_env(self.env)
        return self._update_buf(obs)

    def __getattr__(self, name: str):
        return getattr(self.env, name)


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained PPO agent play Snake (pygame).")
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--which", type=str, default="auto", choices=["auto", "latest", "best", "final"])
    p.add_argument("--reload", type=float, default=0.0, help="If >0, poll for newer checkpoint every N seconds.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--pixel-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--debug-steps", type=int, default=60, help="Print per-step debug for first N steps.")
    return p.parse_args()


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root(Path.cwd())
    run_dir = _resolve_run_dir(repo_root, args.run)

    cfg_d = _load_run_cfg_minimal(run_dir)

    current_ckpt = _pick_checkpoint(run_dir=run_dir, which=args.which)
    current_mtime = current_ckpt.stat().st_mtime
    model = _load_model(current_ckpt, device=str(args.device))

    game = _make_game_from_cfg_d(cfg_d, seed=int(args.seed))
    env_cls = _get_env_cls_from_cfg_d(cfg_d)

    env_params = dict(cast(dict[str, Any], cfg_d["env"]).get("params", {}))
    env = env_cls(game, **env_params)
    env.reset(seed=int(args.seed))

    # ---- frame stacking (watch mirrors training config) ----
    n_stack = _get_n_stack_from_cfg_d(cfg_d)
    if n_stack > 1:
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Box):
            env = BoxFrameStackWrapper(env, n_stack=n_stack)

    # ---- startup debug ----
    print("[watch][dbg] run_dir:", run_dir, flush=True)
    print("[watch][dbg] checkpoint:", current_ckpt, flush=True)
    print("[watch][dbg] env cfg:", cfg_d.get("env"), flush=True)
    print("[watch][dbg] observation cfg:", cfg_d.get("observation"), flush=True)
    print("[watch][dbg] n_stack:", n_stack, flush=True)
    print("[watch][dbg] env.observation_space:", getattr(env, "observation_space", None), flush=True)
    print("[watch][dbg] policy obs_space:", getattr(model.policy, "observation_space", None), flush=True)
    print("[watch][dbg] RelativeDirection import:", "OK" if RelativeDirection is not None else "MISSING", flush=True)

    step_counter = {"n": 0}

    def action_fn(_game: SnakeGame):
        nonlocal model, current_ckpt, current_mtime

        # Reload can happen mid-episode (intentionally).
        if args.reload and args.reload > 0:
            try:
                chosen = _pick_checkpoint(run_dir=run_dir, which=args.which)
                mtime = chosen.stat().st_mtime
                if chosen != current_ckpt or mtime > current_mtime:
                    model = _load_model(chosen, device=str(args.device))
                    current_ckpt = chosen
                    current_mtime = mtime
                    print(f"[watch] reloaded: {chosen}", flush=True)
            except Exception as e:
                print("[watch][dbg] reload error:", repr(e), flush=True)
            time.sleep(float(args.reload))

        obs = _get_current_obs_from_env(env)
        obs_for_model = sanitize_observation(obs)

        action, _ = model.predict(obs_for_model, deterministic=True)
        a = int(np.asarray(action).item())

        # Convert to SAME semantic as training (RelativeDirection)
        if RelativeDirection is None:
            rel = a  # fallback, but this is likely wrong for pygame loop
        else:
            rel = RelativeDirection(a)

        if step_counter["n"] < int(args.debug_steps):
            try:
                head = getattr(game, "snake")[0]
                head_xy = (int(getattr(head, "x")), int(getattr(head, "y")))
            except Exception:
                head_xy = None

            try:
                d = getattr(game, "direction", None)
                d_val = getattr(d, "value", d)
            except Exception:
                d_val = None

            try:
                score = int(getattr(game, "score", -1))
            except Exception:
                score = -1

            if isinstance(obs, np.ndarray):
                obs_shape = obs.shape
            else:
                obs_shape = {k: np.asarray(v).shape for k, v in cast(dict, obs).items()}

            print(
                f"[watch][dbg] step={step_counter['n']:04d} "
                f"action={a} rel={rel} dir={d_val} head={head_xy} score={score} obs={obs_shape}",
                flush=True,
            )

        step_counter["n"] += 1
        return rel

    run_pygame_app(
        game=game,
        cfg=AppConfig(
            fps=int(args.fps),
            pixel_size=int(args.pixel_size),
            caption=f"Snake (watch: {run_dir.name} / {args.which})",
            enable_human_input=False,
        ),
        action_fn=action_fn,
    )

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
