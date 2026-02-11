# src\snake_rl\game\rendering\pygame\app.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

from snake_rl.game.geometry import RelativeDirection
from snake_rl.game.rendering.pygame.renderer import PygameRenderer
from snake_rl.game.rendering.pygame.window import PygameRenderContext, create_pygame_context
from snake_rl.game.snakegame import MoveResult, SnakeGame

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


_END_MASK: MoveResult = (
    MoveResult.HIT_WALL
    | MoveResult.HIT_SELF
    | MoveResult.HIT_BOUNDARY
    | MoveResult.GAME_NOT_RUNNING
    | MoveResult.TIMEOUT
    | MoveResult.WIN
)


def run_pygame_app(
    *,
    game: SnakeGame,
    cfg: AppConfig,
    step_fn: StepFn | None = None,
) -> None:
    """
    pygame UI loop.

    Modes:
      - step_fn mode (preferred for RL watch): the caller advances the environment
        by exactly one step inside step_fn(); this function does NOT call game.move().
      - internal stepping (human): computes RelativeDirection from buffered input
        and calls game.move().
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
                    # Human mode: pygame loop steps the game directly.
                    if game.running:
                        if cfg.enable_human_input:
                            rel = (
                                queued_turn
                                if queued_turn is not None
                                else RelativeDirection.FORWARD
                            )
                            queued_turn = None  # consume once per step
                        else:
                            rel = RelativeDirection.FORWARD

                        results = game.move(rel)
                        if cfg.reset_on_done and (results & _END_MASK):
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
