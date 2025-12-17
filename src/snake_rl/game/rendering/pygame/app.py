# src/snake_rl/game/rendering/app.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from snake_rl.game.geometry import RelativeDirection
from snake_rl.game.rendering.pygame.renderer import PygameRenderer
from snake_rl.game.rendering.pygame.window import PygameRenderContext, create_pygame_context
from snake_rl.game.snakegame import MoveResult, SnakeGame

ActionFn = Callable[[SnakeGame], RelativeDirection]


@dataclass(slots=True)
class AppConfig:
    fps: int
    pixel_size: int = 10
    caption: str = "Snake"
    reset_on_done: bool = True
    # Human input (optional)
    enable_human_input: bool = False
    turn_keys: tuple[int, int] = (pygame.K_a, pygame.K_d)  # left, right


_END_RESULTS: set[MoveResult] = {
    MoveResult.HIT_WALL,
    MoveResult.HIT_SELF,
    MoveResult.HIT_BOUNDARY,
    MoveResult.GAME_NOT_RUNNING,
    MoveResult.TIMEOUT,
    MoveResult.WIN,
}


def run_pygame_app(*, game: SnakeGame, cfg: AppConfig, action_fn: Optional[ActionFn] = None) -> None:
    pygame.init()
    try:
        ctx: PygameRenderContext = create_pygame_context(
            game=game,
            pixel_size=cfg.pixel_size,
            caption=cfg.caption,
        )
        renderer = PygameRenderer(pixel_size=cfg.pixel_size)

        paused = False
        queued_turn: RelativeDirection | None = None  # buffered human input

        left_key, right_key = cfg.turn_keys

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        game.reset()
                        paused = False
                        queued_turn = None
                    elif cfg.enable_human_input:
                        # Buffer exactly one upcoming turn (latest wins)
                        if event.key == left_key:
                            queued_turn = RelativeDirection.LEFT
                        elif event.key == right_key:
                            queued_turn = RelativeDirection.RIGHT

            if not paused:
                if game.running:
                    if action_fn is not None:
                        rel = action_fn(game)
                    elif cfg.enable_human_input:
                        rel = queued_turn if queued_turn is not None else RelativeDirection.FORWARD
                        queued_turn = None  # consume once per step
                    else:
                        rel = RelativeDirection.FORWARD

                    results = game.move(rel)
                    if cfg.reset_on_done and any(r in _END_RESULTS for r in results):
                        game.reset()
                        queued_turn = None
                else:
                    if cfg.reset_on_done:
                        game.reset()
                        queued_turn = None

            renderer.draw(game=game, ctx=ctx)
            pygame.display.flip()
            ctx.clock.tick(cfg.fps)
    finally:
        pygame.quit()
