# src/snake_rl/cli/play.py
from __future__ import annotations

import argparse

from snake_rl.game.level import EmptyLevel
from snake_rl.game.rendering.pygame.app import AppConfig, run_pygame_app
from snake_rl.game.snakegame import SnakeGame


def parse_args():
    p = argparse.ArgumentParser(description="Snake (human-controlled)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=22)
    p.add_argument("--height", type=int, default=13)
    p.add_argument("--food", type=int, default=1)
    p.add_argument("--pixel-size", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    game = SnakeGame(EmptyLevel(args.height, args.width), food_count=args.food)

    run_pygame_app(
        game=game,
        cfg=AppConfig(
            fps=args.fps,
            pixel_size=args.pixel_size,
            caption="Snake (human)",
            enable_human_input=True,
        ),
        action_fn=None,
    )


if __name__ == "__main__":
    main()
