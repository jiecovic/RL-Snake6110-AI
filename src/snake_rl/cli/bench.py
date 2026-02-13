# src/snake_rl/cli/bench.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces

from snake_rl import _core as core
from snake_rl.config.loader import load_train_config_from_path
from snake_rl.rl.envs.factory import make_vec_env
from snake_rl.utils.runs.paths import repo_root

TERMINAL_MASK = (
    int(core.MOVE_HIT_SELF)
    | int(core.MOVE_HIT_WALL)
    | int(core.MOVE_HIT_BOUNDARY)
    | int(core.MOVE_NOT_RUNNING)
    | int(core.MOVE_WIN)
    | int(core.MOVE_TIMEOUT)
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Rust Snake engines/envs (steps/s).")
    p.add_argument(
        "--mode",
        choices=["engine", "env"],
        default="engine",
        help="Benchmark core engines or full envs from config.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path under configs/ (env mode only).",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable, env mode only).",
    )
    p.add_argument("--width", type=int, default=22)
    p.add_argument("--height", type=int, default=13)
    p.add_argument("--food", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cardinal", action="store_true", help="Use cardinal actions (0..3).")
    p.add_argument("--single-steps", type=int, default=200_000)
    p.add_argument("--vec-steps", type=int, default=50_000)
    p.add_argument("--vec-envs", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional timeout limit (steps since food).",
    )
    p.add_argument("--no-single", action="store_true", help="Skip single-engine bench.")
    p.add_argument("--no-vec", action="store_true", help="Skip vec-engine bench.")
    p.add_argument("--no-reset", action="store_true", help="Do not reset on terminal.")
    return p.parse_args()


def _maybe_reset_single(engine, mask: int, reset_on_done: bool) -> None:
    if not reset_on_done:
        return
    if int(mask) & TERMINAL_MASK:
        engine.reset(None)


def _maybe_reset_vec(vec_engine, masks: np.ndarray, reset_on_done: bool) -> None:
    if not reset_on_done:
        return
    done_idx = np.where((masks & TERMINAL_MASK) != 0)[0]
    for idx in done_idx.tolist():
        vec_engine.reset_one(int(idx), seed=None)


def _bench_single(
    *,
    board,
    food_count: int,
    steps: int,
    warmup: int,
    seed: int,
    cardinal: bool,
    reset_on_done: bool,
    max_steps: int | None,
) -> None:
    if steps <= 0:
        return

    engine = core.SnakeEngine(board=board, food_count=int(food_count), seed=None, frame_stack_n=1)
    engine.reset(None)
    if max_steps is not None:
        engine.set_max_steps(int(max_steps))

    rng = np.random.default_rng(int(seed))
    n_actions = 4 if cardinal else 3
    step_fn = engine.step_cardinal if cardinal else engine.step

    if warmup > 0:
        warm_actions = rng.integers(0, n_actions, size=warmup, dtype=np.int64)
        for a in warm_actions:
            mask = step_fn(int(a))
            _maybe_reset_single(engine, int(mask), reset_on_done)

    actions = rng.integers(0, n_actions, size=steps, dtype=np.int64)
    t0 = time.perf_counter()
    for a in actions:
        mask = step_fn(int(a))
        _maybe_reset_single(engine, int(mask), reset_on_done)
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    sps = float(steps) / elapsed
    print(
        f"single: steps={steps:,} elapsed={elapsed:.3f}s steps/s={sps:,.0f} "
        f"action={'cardinal' if cardinal else 'relative'}"
    )


def _bench_vec(
    *,
    board,
    food_count: int,
    vec_envs: int,
    steps: int,
    warmup: int,
    seed: int,
    cardinal: bool,
    reset_on_done: bool,
    max_steps: int | None,
) -> None:
    if steps <= 0 or vec_envs <= 0:
        return

    vec_engine = core.VecSnakeEngine(
        n=int(vec_envs),
        board=board,
        food_count=int(food_count),
        seeds=None,
        frame_stack_n=1,
    )
    if max_steps is not None:
        vec_engine.set_max_steps(int(max_steps))

    rng = np.random.default_rng(int(seed) + 1)
    n_actions = 4 if cardinal else 3
    step_fn = vec_engine.step_cardinal if cardinal else vec_engine.step

    if warmup > 0:
        warm_actions = rng.integers(0, n_actions, size=(warmup, vec_envs), dtype=np.uint8)
        for row in warm_actions:
            masks = np.asarray(step_fn(row.tolist()), dtype=np.uint32)
            _maybe_reset_vec(vec_engine, masks, reset_on_done)

    actions = rng.integers(0, n_actions, size=(steps, vec_envs), dtype=np.uint8)
    t0 = time.perf_counter()
    for row in actions:
        masks = np.asarray(step_fn(row.tolist()), dtype=np.uint32)
        _maybe_reset_vec(vec_engine, masks, reset_on_done)
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    total_steps = int(steps) * int(vec_envs)
    sps = float(total_steps) / elapsed
    print(
        f"vec: envs={vec_envs} steps={steps:,} total={total_steps:,} "
        f"elapsed={elapsed:.3f}s steps/s={sps:,.0f} "
        f"action={'cardinal' if cardinal else 'relative'}"
    )


def _resolve_config_path(config: str | None) -> Path:
    if not config:
        raise ValueError("--config is required in env mode")
    repo = repo_root()
    config_root = repo / "configs"

    path = Path(config)
    if not path.is_file():
        candidate = config_root / path
        if candidate.is_file():
            path = candidate

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    if not path.resolve().is_relative_to(config_root.resolve()):
        raise ValueError("--config must be under configs/")

    return path


def _bench_env(*, config: str | None, overrides: list[str], steps: int, warmup: int, seed: int):
    config_path = _resolve_config_path(config)
    repo = repo_root()
    config_root = repo / "configs"
    cfg, _, _, _ = load_train_config_from_path(
        config_path=config_path,
        config_root=config_root,
        overrides=list(overrides),
    )

    vec_env = make_vec_env(cfg=cfg)
    try:
        action_space = vec_env.action_space
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("Only Discrete action spaces are supported for benchmark.")
        n_actions = int(action_space.n)
        num_envs = int(getattr(vec_env, "num_envs", 1))

        _ = vec_env.reset()

        rng = np.random.default_rng(int(seed))
        if warmup > 0:
            warm_actions = rng.integers(0, n_actions, size=(warmup, num_envs), dtype=np.int64)
            for row in warm_actions:
                vec_env.step(row)

        actions = rng.integers(0, n_actions, size=(steps, num_envs), dtype=np.int64)
        t0 = time.perf_counter()
        for row in actions:
            vec_env.step(row)
        t1 = time.perf_counter()

        elapsed = max(1e-9, t1 - t0)
        total_steps = int(steps) * int(num_envs)
        sps = float(total_steps) / elapsed
        print(
            f"env: vec={cfg.run.vec} envs={num_envs} steps={steps:,} total={total_steps:,} "
            f"elapsed={elapsed:.3f}s steps/s={sps:,.0f} obs={cfg.env.obs.kind}/{cfg.env.obs.view}"
        )
    finally:
        vec_env.close()


def main() -> None:
    args = _parse_args()

    if str(args.mode).lower() == "env":
        _bench_env(
            config=args.config,
            overrides=list(args.override),
            steps=int(args.vec_steps),
            warmup=int(args.warmup),
            seed=int(args.seed),
        )
        return

    board = core.Board(width=int(args.width), height=int(args.height))

    reset_on_done = not bool(args.no_reset)
    max_steps = args.max_steps if args.max_steps is None else int(args.max_steps)

    if not bool(args.no_single):
        _bench_single(
            board=board,
            food_count=int(args.food),
            steps=int(args.single_steps),
            warmup=int(args.warmup),
            seed=int(args.seed),
            cardinal=bool(args.cardinal),
            reset_on_done=reset_on_done,
            max_steps=max_steps,
        )

    if not bool(args.no_vec):
        _bench_vec(
            board=board,
            food_count=int(args.food),
            vec_envs=int(args.vec_envs),
            steps=int(args.vec_steps),
            warmup=int(args.warmup),
            seed=int(args.seed),
            cardinal=bool(args.cardinal),
            reset_on_done=reset_on_done,
            max_steps=max_steps,
        )


if __name__ == "__main__":
    main()
