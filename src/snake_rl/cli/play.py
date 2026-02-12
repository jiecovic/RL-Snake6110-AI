# src/snake_rl/cli/play.py
from __future__ import annotations

import argparse

from snake_rl import _core as core
from snake_rl.envs.specs import ObservationSpec
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snake_engine import SnakeEngine


def parse_args():
    p = argparse.ArgumentParser(description="Snake (human-controlled)")
    p.add_argument("--fps", type=int, default=0, help="Render FPS cap (0 = uncapped).")
    p.add_argument("--sim-hz", type=int, default=10, help="Simulation steps per second.")
    p.add_argument("--width", type=int, default=22)
    p.add_argument("--height", type=int, default=13)
    p.add_argument("--food", type=int, default=1)
    p.add_argument("--pixel-size", type=int, default=10)
    p.add_argument("--agent-view-radius", type=int, default=6)
    p.add_argument("--no-agent-view", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    board = core.Board(width=int(args.width), height=int(args.height))
    game = SnakeEngine(board=board, food_count=args.food)

    agent_view_spec = None
    if not bool(args.no_agent_view):
        agent_view_spec = ObservationSpec(
            kind="pixel",
            view="head",
            params={
                "view_radius": int(args.agent_view_radius),
                "rotate_to_head": True,
            },
        )

    run_pygame_app(
        game=game,
        cfg=AppConfig(
            fps=args.fps,
            sim_hz=args.sim_hz,
            pixel_size=args.pixel_size,
            caption="Snake (human)",
            enable_human_input=True,
            agent_view_spec=agent_view_spec,
        ),
    )


if __name__ == "__main__":
    main()
