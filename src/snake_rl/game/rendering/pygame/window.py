# src/snake_rl/game/rendering/pygame/window.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from snake_rl.game.snake_engine import SnakeEngine

try:
    import pygame as _pygame
except Exception:  # pragma: no cover
    _pygame = None
pygame: Any = _pygame


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))


@dataclass(slots=True)
class LayoutConfig:
    margin: int = 14
    panel_gap: int = 14
    panel_padding: int = 10
    panel_label_height: int = 20
    hud_height: int = 92
    hud_padding_x: int = 14
    hud_padding_y: int = 10
    font_name: str = "Consolas"
    font_size: int = 18
    label_font_size: int = 16


@dataclass(slots=True)
class PygameRenderContext:
    screen: Any
    clock: Any
    font: Any
    label_font: Any

    world_panel: Rect
    world_view: Rect
    agent_panel: Rect | None
    agent_view: Rect | None
    hud_panel: Rect

    layout: LayoutConfig
    sim_fps: float = 0.0
    sim_steps: int = 0
    paused: bool = False
    target_sim_hz: int = 0


def create_pygame_context(
    *,
    game: SnakeEngine,
    pixel_size: int,
    caption: str,
    agent_view_grid: tuple[int, int] | None = None,
    layout: LayoutConfig | None = None,
) -> PygameRenderContext:
    if pygame is None:  # pragma: no cover
        raise RuntimeError("pygame is not installed")

    cfg = layout if layout is not None else LayoutConfig()

    # pygame.init() is owned by run_pygame_app (app.py)
    pygame.display.set_caption(caption)

    tile_dim = int(game.tile_size) * int(pixel_size)
    world_w = int(game.width) * tile_dim
    world_h = int(game.height) * tile_dim

    panel_pad = int(cfg.panel_padding)
    label_h = int(cfg.panel_label_height)
    gap = int(cfg.panel_gap)
    margin = int(cfg.margin)

    world_panel_w = world_w + 2 * panel_pad
    world_panel_h = world_h + 2 * panel_pad + label_h

    agent_panel_w = 0
    agent_panel_h = 0
    agent_view_w = 0
    agent_view_h = 0
    if agent_view_grid is not None:
        agent_view_w = int(agent_view_grid[0]) * int(pixel_size)
        agent_view_h = int(agent_view_grid[1]) * int(pixel_size)
        agent_panel_w = agent_view_w + 2 * panel_pad
        agent_panel_h = agent_view_h + 2 * panel_pad + label_h

    row_h = max(world_panel_h, agent_panel_h)

    win_w = margin * 2 + world_panel_w + (gap + agent_panel_w if agent_panel_w else 0)
    win_h = margin * 2 + row_h + gap + int(cfg.hud_height)

    screen = pygame.display.set_mode((int(win_w), int(win_h)))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(cfg.font_name, int(cfg.font_size))
    label_font = pygame.font.SysFont(cfg.font_name, int(cfg.label_font_size))

    world_panel = Rect(x=margin, y=margin, w=world_panel_w, h=world_panel_h)
    world_view = Rect(
        x=world_panel.x + panel_pad,
        y=world_panel.y + panel_pad + label_h,
        w=world_w,
        h=world_h,
    )

    agent_panel: Rect | None = None
    agent_view: Rect | None = None
    if agent_panel_w and agent_panel_h:
        ax = margin + world_panel_w + gap
        agent_panel = Rect(x=ax, y=margin, w=agent_panel_w, h=agent_panel_h)
        agent_view = Rect(
            x=agent_panel.x + panel_pad,
            y=agent_panel.y + panel_pad + label_h,
            w=agent_view_w,
            h=agent_view_h,
        )

    hud_panel = Rect(
        x=margin,
        y=margin + row_h + gap,
        w=win_w - 2 * margin,
        h=int(cfg.hud_height),
    )

    return PygameRenderContext(
        screen=screen,
        clock=clock,
        font=font,
        label_font=label_font,
        world_panel=world_panel,
        world_view=world_view,
        agent_panel=agent_panel,
        agent_view=agent_view,
        hud_panel=hud_panel,
        layout=cfg,
    )
