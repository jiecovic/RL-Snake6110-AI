# src/snake_rl/game/rendering/pygame/window.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snake_rl.game.snakegame import SnakeGame

try:
    import pygame as _pygame
except Exception:  # pragma: no cover
    _pygame = None
pygame: Any = _pygame


@dataclass(slots=True)
class PygameRenderContext:
    screen: Any
    clock: Any
    font: Any

    hud_height: int
    hud_padding_x: int
    hud_padding_y: int


def create_pygame_context(
    *,
    game: SnakeGame,
    pixel_size: int,
    caption: str,
    hud_height: int = 26,
    hud_padding_x: int = 10,
    hud_padding_y: int = 4,
    font_name: str = "Consolas",
    font_size: int = 18,
) -> PygameRenderContext:
    if pygame is None:  # pragma: no cover
        raise RuntimeError("pygame is not installed")

    # pygame.init() is owned by run_pygame_app (app.py)
    pygame.display.set_caption(caption)

    tile_dim = game.tile_size * pixel_size
    win_w = game.width * tile_dim
    win_h = game.height * tile_dim + hud_height

    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(font_name, font_size)

    return PygameRenderContext(
        screen=screen,
        clock=clock,
        font=font,
        hud_height=hud_height,
        hud_padding_x=hud_padding_x,
        hud_padding_y=hud_padding_y,
    )
