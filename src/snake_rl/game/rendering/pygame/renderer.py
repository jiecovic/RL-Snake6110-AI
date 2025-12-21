# src/snake_rl/game/rendering/pygame/renderer.py
from __future__ import annotations

import pygame

from snake_rl.game.rendering.pygame.surf import gray255_to_surface
from snake_rl.game.rendering.pygame.window import PygameRenderContext
from snake_rl.game.snakegame import SnakeGame


class PygameRenderer:
    def __init__(self, *, pixel_size: int = 10):
        self.pixel_size = int(pixel_size)

    def draw(self, *, game: SnakeGame, ctx: PygameRenderContext) -> None:
        if game.pixel_buffer is None or game.pixel_buffer.size == 0:
            return

        ctx.screen.fill((0, 0, 0))

        # Board: pixel_buffer is now [0,255] uint8
        surface = gray255_to_surface(game.pixel_buffer, pixel_size=self.pixel_size)
        ctx.screen.blit(surface, (0, ctx.hud_height))

        # HUD
        score_text = f"Score: {game.score}"
        fps_text = f"FPS: {ctx.clock.get_fps():.1f}"
        empty_text = f"Empty: {len(game.spawnable_tiles)}"
        combined = f"{score_text}    {fps_text}    {empty_text}"

        info_surf = ctx.font.render(combined, True, (230, 230, 230))
        ctx.screen.blit(info_surf, (ctx.hud_padding_x, ctx.hud_padding_y))
