import pygame
import sys
from typing import Optional

from core.snake6110.geometry import RelativeDirection
from core.snake6110.level import LevelFromTemplate
from core.snake6110.snakegame import SnakeGame  # Updated to match your current class


def handle_turn_input(key: int) -> Optional[RelativeDirection]:
    if key == pygame.K_a:
        return RelativeDirection.LEFT
    elif key == pygame.K_d:
        return RelativeDirection.RIGHT
    return None


def main():
    pygame.init()

    # Replace this with your actual level loading logic
    level = LevelFromTemplate()
    game = SnakeGame(level, food_count=1, fps=10, tile_size=32)

    print(f"Initial snake: {game.snake}")
    print(f"Initial direction: {game.direction}")
    print(f"Initial food: {game.food}")

    while game.running:
        relative_dir = RelativeDirection.FORWARD

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                in_dir = handle_turn_input(event.key)
                if in_dir is not None:
                    relative_dir = in_dir

        game.move(relative_dir)
        game.render()

    pygame.quit()
    print("Game Over!")


if __name__ == "__main__":
    main()
