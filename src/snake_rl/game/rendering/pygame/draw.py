# src/snake_rl/game/rendering/pygame/draw.py

from __future__ import annotations

import numpy as np

from snake_rl.game.rendering.pygame.hud import format_pairs
from snake_rl.game.rendering.pygame.theme import ACCENT, MUTED, PANEL_BG, PANEL_BORDER, TEXT
from snake_rl.game.rendering.pygame.window import PygameRenderContext, Rect


def draw_panel(
    *,
    ctx: PygameRenderContext,
    panel: Rect,
    title: str,
    subtitle: str | None = None,
) -> None:
    ctx.screen.fill(PANEL_BG, panel.to_tuple())
    ctx.screen.fill(PANEL_BORDER, panel.to_tuple(), 1)

    tx = int(panel.x + ctx.layout.panel_padding)
    ty = int(panel.y + 3)
    title_surf = ctx.label_font.render(str(title), True, ACCENT)
    ctx.screen.blit(title_surf, (tx, ty))

    if subtitle:
        sub_surf = ctx.label_font.render(str(subtitle), True, MUTED)
        ctx.screen.blit(sub_surf, (tx + title_surf.get_width() + 10, ty))


def draw_hud_box(
    *,
    ctx: PygameRenderContext,
    box: Rect,
    title: str,
    pairs: list[tuple[str, str]],
    empty_text: str | None = None,
    border_mask: set[str] | None = None,
) -> None:
    ctx.screen.fill(PANEL_BG, box.to_tuple())
    if border_mask is None:
        ctx.screen.fill(PANEL_BORDER, box.to_tuple(), 1)
    else:
        x, y, w, h = box.to_tuple()
        if "top" in border_mask:
            ctx.screen.fill(PANEL_BORDER, (x, y, w, 1))
        if "bottom" in border_mask:
            ctx.screen.fill(PANEL_BORDER, (x, y + h - 1, w, 1))
        if "left" in border_mask:
            ctx.screen.fill(PANEL_BORDER, (x, y, 1, h))
        if "right" in border_mask:
            ctx.screen.fill(PANEL_BORDER, (x + w - 1, y, 1, h))

    tx = int(box.x + 8)
    ty = int(box.y + 4)
    title_surf = ctx.label_font.render(str(title), True, ACCENT)
    ctx.screen.blit(title_surf, (tx, ty))

    x = int(box.x + 8)
    y = int(box.y + 24)
    dy = ctx.label_font.get_height() + 4

    lines = format_pairs(pairs, empty_text=empty_text, key_pad=0)
    for i, line in enumerate(lines):
        ctx.screen.blit(ctx.label_font.render(line, True, TEXT), (x, y + i * dy))


def draw_hud_controls(*, ctx: PygameRenderContext, text: str) -> None:
    hud = ctx.hud_panel
    x = int(hud.x + ctx.layout.hud_padding_x)
    y = int(hud.y + hud.h - ctx.layout.hud_padding_y - ctx.label_font.get_height())
    ctx.screen.blit(ctx.label_font.render(text, True, MUTED), (x, y))


def categorical_to_gray_pixels(
    frame: np.ndarray,
    *,
    num_classes: int,
    tile_size: int,
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise TypeError(f"categorical frame must be 2D, got {arr.shape}")
    k = max(1, int(num_classes))
    if k <= 1:
        gray = np.zeros_like(arr, dtype=np.uint8)
    else:
        gray_f = (arr.astype(np.float32) / float(k - 1)) * 255.0
        gray = gray_f.astype(np.uint8, copy=False)
    if int(tile_size) <= 1:
        return gray
    ts = int(tile_size)
    return np.repeat(np.repeat(gray, ts, axis=0), ts, axis=1)
