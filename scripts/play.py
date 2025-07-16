import pygame
import sys
import argparse
from typing import Optional

from core.snake6110.geometry import RelativeDirection
from core.snake6110.level import EmptyLevel
from core.snake6110.snakegame import SnakeGame, MoveResult


def handle_turn_input(key: int) -> Optional[RelativeDirection]:
    if key == pygame.K_a:
        return RelativeDirection.LEFT
    elif key == pygame.K_d:
        return RelativeDirection.RIGHT
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="SnakeGame interactive runner")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--width", type=int, default=22, help="Grid width")
    parser.add_argument("--height", type=int, default=13, help="Grid height")
    parser.add_argument("--food", type=int, default=1, help="Number of food items on the board")
    return parser.parse_args()


def main():
    args = parse_args()
    pygame.init()

    level = EmptyLevel(args.height, args.width)
    game = SnakeGame(level, food_count=args.food, fps=args.fps)
    paused = False

    while True:
        relative_dir = RelativeDirection.FORWARD

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Unpaused")

                elif event.key == pygame.K_r:
                    game.reset()
                    paused = False
                    print("Manual reset.")

                elif not paused:
                    in_dir = handle_turn_input(event.key)
                    if in_dir is not None:
                        relative_dir = in_dir

        if not paused:
            if game.running:
                result = game.move(relative_dir)
                game.render()

                for res in result:
                    if res in (
                            MoveResult.HIT_WALL,
                            MoveResult.HIT_SELF,
                            MoveResult.HIT_BOUNDARY,
                            MoveResult.GAME_NOT_RUNNING,
                            MoveResult.TIMEOUT,
                            MoveResult.WIN,
                    ):
                        print(f"[END] {res.name}")
                        break
            else:
                game.render()
                game.reset()


if __name__ == "__main__":
    main()
