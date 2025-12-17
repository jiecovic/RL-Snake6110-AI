# src/snake_rl/game/rendering/renderer.py
from __future__ import annotations

import numpy as np
import pygame

from snake_rl.game.rendering.pygame.window import PygameRenderContext
from snake_rl.game.snakegame import SnakeGame


class PygameRenderer:
    def __init__(self, *, pixel_size: int = 10):
        self.pixel_size = pixel_size

    def draw(self, *, game: SnakeGame, ctx: PygameRenderContext) -> None:
        if game.pixel_buffer is None or game.pixel_buffer.size == 0:
            return

        ctx.screen.fill((0, 0, 0))

        # Board
        upscaled = np.kron(
            game.pixel_buffer,
            np.ones((self.pixel_size, self.pixel_size), dtype=game.pixel_buffer.dtype),
        )
        rgb_buffer = np.stack([upscaled * 255] * 3, axis=-1).astype(np.uint8)
        surface = pygame.surfarray.make_surface(rgb_buffer.swapaxes(0, 1))
        ctx.screen.blit(surface, (0, ctx.hud_height))

        # HUD
        score_text = f"Score: {game.score}"
        fps_text = f"FPS: {ctx.clock.get_fps():.1f}"
        empty_text = f"Empty: {len(game.spawnable_tiles)}"
        combined = f"{score_text}    {fps_text}    {empty_text}"

        info_surf = ctx.font.render(combined, True, (230, 230, 230))
        ctx.screen.blit(info_surf, (ctx.hud_padding_x, ctx.hud_padding_y))
