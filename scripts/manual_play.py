import pygame
import sys
from typing import Optional

from core.snake6110.geometry import RelativeDirection
from core.snake6110.level import LevelFromTemplate, EmptyLevel
from core.snake6110.snakegame import SnakeGame
from core.snake6110.snakegame import MoveResult  # <-- Import this for result checking


def handle_turn_input(key: int) -> Optional[RelativeDirection]:
    if key == pygame.K_a:
        return RelativeDirection.LEFT
    elif key == pygame.K_d:
        return RelativeDirection.RIGHT
    return None


def main():
    pygame.init()

    level = EmptyLevel(11, 20)
    game = SnakeGame(level, food_count=1, fps=10)
    paused = False

    # --- FPS tracking ---
    frame_count = 0
    last_fps_time = pygame.time.get_ticks()

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

                # --- Count frame ---
                frame_count += 1
                now = pygame.time.get_ticks()
                elapsed_ms = now - last_fps_time
                if elapsed_ms >= 1000:
                    fps = frame_count / (elapsed_ms / 1000.0)
                    print(f"[FPS] {fps:.2f}")
                    frame_count = 0
                    last_fps_time = now

                if result == MoveResult.HIT_WALL:
                    print("Game over: hit wall.")
                elif result == MoveResult.HIT_SELF:
                    print("Game over: hit itself.")
                elif result == MoveResult.HIT_BOUNDARY:
                    print("Game over: hit boundary.")
            else:
                game.render()
                game.reset()


if __name__ == "__main__":
    main()
