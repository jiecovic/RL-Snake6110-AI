# src/snake_rl/game/rendering/pygame/app.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import snake_rl._core as core

from snake_rl.game.rendering.pygame.renderer import PygameRenderer
from snake_rl.game.rendering.pygame.window import PygameRenderContext, create_pygame_context
from snake_rl.game.snake_engine import SnakeEngine

try:
    import pygame as _pygame
except Exception:  # pragma: no cover
    _pygame = None

pygame: Any = _pygame

StepFn = Callable[[], None]


@dataclass(slots=True)
class AppConfig:
    fps: int
    pixel_size: int = 10
    caption: str = "Snake"
    reset_on_done: bool = True
    # Human input (optional)
    enable_human_input: bool = False
    turn_keys: tuple[int, int] | None = None  # left, right


_END_MASK: int = (
    core.MOVE_HIT_WALL
    | core.MOVE_HIT_SELF
    | core.MOVE_HIT_BOUNDARY
    | core.MOVE_NOT_RUNNING
    | core.MOVE_TIMEOUT
    | core.MOVE_WIN
)


def run_pygame_app(
    *,
    game: SnakeEngine,
    cfg: AppConfig,
    step_fn: StepFn | None = None,
) -> None:
    """
    pygame UI loop.

    Modes:
      - step_fn mode (preferred for RL watch): the caller advances the environment
        by exactly one step inside step_fn(); this function does NOT call game.move().
      - internal stepping (human): uses relative action ints from buffered input
        and calls game.move().
    """
    if pygame is None:  # pragma: no cover
        raise RuntimeError("pygame is not installed")

    pygame.init()
    try:
        ctx: PygameRenderContext = create_pygame_context(
            game=game,
            pixel_size=cfg.pixel_size,
            caption=cfg.caption,
        )
        renderer = PygameRenderer(pixel_size=cfg.pixel_size)

        paused = False
        queued_turn: int | None = None  # buffered human input

        if cfg.turn_keys is None:
            left_key, right_key = (pygame.K_a, pygame.K_d)
        else:
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
                            queued_turn = 1
                        elif event.key == right_key:
                            queued_turn = 2

            if not paused:
                if step_fn is not None:
                    # RL-consistent mode: caller owns stepping (env.step()).
                    step_fn()
                else:
                    # Human mode: pygame loop steps the game directly.
                    if game.running:
                        if cfg.enable_human_input:
                            rel = queued_turn if queued_turn is not None else 0
                            queued_turn = None  # consume once per step
                        else:
                            rel = 0

                        results = game.move(int(rel))
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
