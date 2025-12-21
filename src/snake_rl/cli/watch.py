# src/snake_rl/cli/watch.py
from __future__ import annotations

import argparse
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
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
from snake_rl.tools.agent_view_stream import AgentViewStream
from snake_rl.tools.obs_debug import debug_print_obs
from snake_rl.training.env_factory import apply_frame_stack
from snake_rl.utils.checkpoints import pick_checkpoint
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.model_params import format_sb3_param_report, format_sb3_param_summary
from snake_rl.utils.models import load_ppo
from snake_rl.utils.obs import sanitize_observation
from snake_rl.utils.obs_render import obs_frame_to_pixels
from snake_rl.utils.paths import relpath, repo_root, resolve_run_dir

try:
    # Optional dependency: only needed for tile_vocab runs.
    from snake_rl.vocab import load_tile_vocab
except Exception:  # pragma: no cover
    load_tile_vocab = None  # type: ignore[assignment]


def _ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _make_game_from_level_params(level: dict[str, int], *, seed: int) -> SnakeGame:
    lvl = EmptyLevel(height=int(level["height"]), width=int(level["width"]))
    return SnakeGame(level=lvl, food_count=int(level["food_count"]), seed=int(seed))


def _filter_env_kwargs(*, env_cls: type, params: dict[str, Any]) -> dict[str, Any]:
    """
    Filter snapshot env params to only those accepted by env_cls.__init__.

    This is important because snapshots may contain reproducibility metadata
    (e.g. tile_vocab_meta) that should NOT be passed to env constructors.
    """
    if not params:
        return {}

    try:
        sig = inspect.signature(env_cls.__init__)
    except Exception:
        # Best-effort fallback: drop known meta keys.
        drop = {"tile_vocab_meta"}
        return {k: v for k, v in params.items() if k not in drop}

    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    # If env supports **kwargs, keep everything.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)

    filtered = {k: v for k, v in params.items() if k in accepted}
    return filtered


def _infer_num_classes_from_obs_space(obs_space: spaces.Space) -> Optional[int]:
    """
    For tile/class-id Box spaces, infer K from high=max_id => K=max_id+1.
    Supports Box directly or Dict{"tiles": Box} (future-proof).
    """
    box: Optional[spaces.Box] = None
    if isinstance(obs_space, spaces.Box):
        box = obs_space
    elif isinstance(obs_space, spaces.Dict) and "tiles" in obs_space.spaces:
        sub = obs_space.spaces["tiles"]
        if isinstance(sub, spaces.Box):
            box = sub

    if box is None:
        return None

    hi = box.high
    try:
        max_hi = float(np.asarray(hi).max())
    except Exception:
        return None
    return int(max_hi) + 1


def _obs_has_pixel_key(obs: Any, *, pixel_key: str = "pixel") -> bool:
    return isinstance(obs, dict) and (pixel_key in obs)


def _extract_tile_grid_2d(obs: Any) -> Optional[np.ndarray]:
    """
    Convert the current observation into a 2D uint8 grid of class/tile ids.

    Supported shapes:
      - Box obs from VecEnv: np.ndarray with shape (n_envs, C, H, W)
      - Box obs without channel: (n_envs, H, W)

    We always take env index 0 (watch uses DummyVecEnv with 1 env),
    and if C>1 (frame stack), we take the LAST channel as "latest frame".
    """
    if not isinstance(obs, np.ndarray):
        return None

    if obs.ndim == 4:
        g = obs[0]  # (C,H,W)
        if g.shape[0] < 1:
            return None
        g2 = g[-1]  # latest stacked frame
        return np.asarray(g2, dtype=np.uint8)
    if obs.ndim == 3:
        g2 = obs[0]  # (H,W)
        return np.asarray(g2, dtype=np.uint8)
    return None


def _load_num_classes_from_env_params(env_params: dict[str, Any]) -> Optional[int]:
    """
    If this run used a named tile_vocab, prefer using its num_classes (true K)
    instead of inferring from observation_space.high (which should match, but
    this is more explicit and gives us class_names too, later).
    """
    name = env_params.get("tile_vocab", None)
    if name is None:
        return None
    if load_tile_vocab is None:
        return None
    try:
        vocab = load_tile_vocab(str(name))
        return int(vocab.num_classes)
    except Exception:
        return None


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

    # Agent-view window
    p.add_argument("--show-agent-view", action="store_true", help="Open a separate window that renders the agent's observation.")
    p.add_argument("--agent-view-max-size", type=int, default=1080, help="Max window side in screen pixels (auto scale).")
    p.add_argument("--agent-view-fps", type=int, default=0, help="If >0, cap agent-view updates to this FPS. 0 => send every tick.")
    p.add_argument("--agent-view-keep-stderr", action="store_true", help="Keep agent-view subprocess stderr (useful for debugging).")
    p.add_argument(
        "--agent-view-pixel-size",
        type=int,
        default=0,
        help="If >0, force pixel_size for agent-view window. 0 => auto. Defaults to --pixel-size if unset.",
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
                self.logger.info(f"reloaded checkpoint: {relpath(chosen, base=self.repo)} (mtime={int(mtime)})")
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
    env_kwargs = _filter_env_kwargs(env_cls=env_cls, params=env_params)

    # (Optional) log dropped keys once (helps catch future metadata additions)
    dropped = sorted(set(env_params.keys()) - set(env_kwargs.keys()))
    if dropped:
        logger.info(f"[watch] ignoring non-constructor env params: {dropped}")

    base_env = env_cls(game, **env_kwargs)  # type: ignore[arg-type]

    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecMonitor(vec_env)

    n_stack = int(get_frame_stack_n(cfg))
    vec_env = apply_frame_stack(vec_env=vec_env, n_stack=n_stack, pixel_key="pixel")

    # Precompute num_classes if possible (used by agent-view ids mode)
    num_classes_from_vocab = _load_num_classes_from_env_params(env_params)
    num_classes_from_space = _infer_num_classes_from_obs_space(vec_env.observation_space)
    num_classes = num_classes_from_vocab or num_classes_from_space

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

    agent_view = AgentViewStream()
    if args.show_agent_view:
        av_ps = int(args.agent_view_pixel_size)
        if av_ps <= 0:
            av_ps = int(args.pixel_size)

        agent_view.start(
            caption="Snake (agent view)",
            max_size=int(args.agent_view_max_size),
            fps=int(args.agent_view_fps),
            keep_stderr=bool(args.agent_view_keep_stderr),
            pixel_size=av_ps,
        )

    def _controller_step() -> None:
        controller.step()

        if not agent_view.is_alive():
            return

        try:
            # Case A: Dict obs with pixels (pixel envs)
            if _obs_has_pixel_key(controller.obs, pixel_key="pixel"):
                frame = obs_frame_to_pixels(
                    controller.obs,
                    tileset=game.tileset,
                    pixel_key="pixel",
                )
                agent_view.send_frame(
                    frame,
                    mode="gray255",
                    max_fps=int(args.agent_view_fps),
                )
                return

            # Case B: Box obs (tile/class-id envs)
            grid = _extract_tile_grid_2d(controller.obs)
            if grid is not None:
                k = int(num_classes) if num_classes is not None else (int(grid.max()) + 1)
                agent_view.send_frame(
                    grid,
                    mode="ids",
                    num_classes=int(k),
                    max_fps=int(args.agent_view_fps),
                )
                return

        except Exception:
            pass

    try:
        run_pygame_app(
            game=game,
            cfg=AppConfig(
                fps=int(args.fps),
                pixel_size=int(args.pixel_size),
                caption=f"Snake (watch: {run_dir.name} / {args.which})",
                enable_human_input=False,
            ),
            step_fn=_controller_step,
        )
    finally:
        try:
            agent_view.close()
        except Exception:
            pass
        vec_env.close()


if __name__ == "__main__":
    main()
