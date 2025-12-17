# src/snake_rl/game/rendering/pygame/app.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from snake_rl.game.geometry import RelativeDirection
from snake_rl.game.rendering.pygame.renderer import PygameRenderer
from snake_rl.game.rendering.pygame.window import PygameRenderContext, create_pygame_context
from snake_rl.game.snakegame import MoveResult, SnakeGame

ActionFn = Callable[[SnakeGame], RelativeDirection]
StepFn = Callable[[], None]


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


def run_pygame_app(
        *,
        game: SnakeGame,
        cfg: AppConfig,
        action_fn: Optional[ActionFn] = None,
        step_fn: Optional[StepFn] = None,
) -> None:
    """
    pygame UI loop.

    Two modes:

    1) step_fn mode (preferred for RL watch):
       - The caller advances the environment by exactly one step inside step_fn().
       - This function does NOT call game.move().

    2) legacy action_fn/human mode (used by play.py):
       - This function computes a RelativeDirection (AI or human) and calls game.move().
    """
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
                        # NOTE: In step_fn mode, resetting only the game can desync env state.
                        # We keep this for legacy/human mode. For watch, don't press R.
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
                if step_fn is not None:
                    # RL-consistent mode: caller owns stepping (env.step()).
                    step_fn()
                else:
                    # Legacy mode: pygame loop steps the game directly.
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
