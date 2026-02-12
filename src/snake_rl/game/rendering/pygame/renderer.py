# src/snake_rl/game/rendering/pygame/renderer.py
from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl.envs.specs import ObservationSpec
from snake_rl.game.rendering.pygame.surf import gray255_to_surface
from snake_rl.game.rendering.pygame.window import PygameRenderContext, Rect
from snake_rl.game.snake_engine import SnakeEngine
from snake_rl.utils.obs_render import categorical_frame_to_pixels
from snake_rl.vocab import TileVocab

_BG = (14, 16, 20)
_PANEL_BG = (23, 27, 34)
_PANEL_BORDER = (60, 70, 86)
_HUD_BG = (19, 22, 28)
_TEXT = (232, 232, 232)
_MUTED = (160, 168, 176)
_ACCENT = (120, 200, 255)


class PygameRenderer:
    def __init__(
        self,
        *,
        pixel_size: int = 10,
        agent_view_spec: ObservationSpec | None = None,
        agent_view_vocab: TileVocab | None = None,
    ):
        self.pixel_size = int(pixel_size)
        self.agent_view_spec = agent_view_spec
        self.agent_view_vocab = agent_view_vocab

    def draw(self, *, game: SnakeEngine, ctx: PygameRenderContext) -> None:
        if game.pixel_buffer is None or game.pixel_buffer.size == 0:
            return

        ctx.screen.fill(_BG)

        self._draw_panel(
            ctx=ctx,
            panel=ctx.world_panel,
            title="WORLD",
            subtitle=f"{int(game.width)}x{int(game.height)} tiles",
        )
        if ctx.agent_panel is not None:
            label = self._agent_label()
            self._draw_panel(ctx=ctx, panel=ctx.agent_panel, title="AGENT", subtitle=label)

        # World view
        world_surface = gray255_to_surface(game.pixel_buffer, pixel_size=self.pixel_size)
        ctx.screen.blit(world_surface, (ctx.world_view.x, ctx.world_view.y))

        # Agent view (optional)
        if ctx.agent_view is not None and self.agent_view_spec is not None:
            frame = self._build_agent_frame(game=game)
            if frame is not None:
                view_surface = self._agent_surface(frame=frame, game=game)
                if view_surface is not None:
                    ctx.screen.blit(view_surface, (ctx.agent_view.x, ctx.agent_view.y))

        # HUD
        self._draw_hud(game=game, ctx=ctx)

    def _draw_panel(
        self,
        *,
        ctx: PygameRenderContext,
        panel: Rect,
        title: str,
        subtitle: str | None = None,
    ) -> None:
        ctx.screen.fill(_PANEL_BG, panel.to_tuple())
        ctx.screen.fill(_PANEL_BORDER, panel.to_tuple(), 1)

        tx = int(panel.x + ctx.layout.panel_padding)
        ty = int(panel.y + 3)
        title_surf = ctx.label_font.render(str(title), True, _ACCENT)
        ctx.screen.blit(title_surf, (tx, ty))

        if subtitle:
            sub_surf = ctx.label_font.render(str(subtitle), True, _MUTED)
            ctx.screen.blit(sub_surf, (tx + title_surf.get_width() + 10, ty))

    def _draw_hud(self, *, game: SnakeEngine, ctx: PygameRenderContext) -> None:
        hud = ctx.hud_panel
        ctx.screen.fill(_HUD_BG, hud.to_tuple())
        ctx.screen.fill(_PANEL_BORDER, hud.to_tuple(), 1)

        fps = float(ctx.clock.get_fps())
        sim = float(ctx.sim_fps)
        steps = int(ctx.sim_steps)
        paused = bool(ctx.paused)
        running = bool(game.running)

        line1 = (
            f"Score {int(game.score)}   "
            f"Len {int(game.snake_len)}   "
            f"Steps {steps}   "
            f"Running {'yes' if running else 'no'}"
        )
        line2 = (
            f"Render {fps:.1f} fps   "
            f"Sim {sim:.1f} fps (target {int(ctx.target_sim_hz)})   "
            f"Empty {int(game.spawnable_count)}"
        )
        if paused:
            line2 = f"{line2}   [PAUSED]"

        x = int(hud.x + ctx.layout.hud_padding_x)
        y = int(hud.y + ctx.layout.hud_padding_y)
        ctx.screen.blit(ctx.font.render(line1, True, _TEXT), (x, y))
        ctx.screen.blit(ctx.font.render(line2, True, _MUTED), (x, y + ctx.font.get_height() + 6))

    def _agent_label(self) -> str:
        if self.agent_view_spec is None:
            return ""
        kind = self.agent_view_spec.kind_norm()
        view = self.agent_view_spec.view_norm()
        vocab = self.agent_view_vocab.name if self.agent_view_vocab is not None else None
        if vocab:
            return f"{kind}/{view} • {vocab}"
        return f"{kind}/{view}"

    def _build_agent_frame(self, *, game: SnakeEngine) -> tuple[np.ndarray, str] | None:
        if self.agent_view_spec is None:
            return None
        spec = self.agent_view_spec
        obs = spec.observe(
            game=game,
            tile_vocab=self.agent_view_vocab,
            initial_snake_length=int(game.snake_len),
            max_playable_tiles=int(game.max_playable_tiles),
        )
        base: Any
        if isinstance(obs, dict):
            key = spec.frame_stack_key() or ("pixel" if spec.kind_norm() == "pixel" else "tiles")
            base = obs.get(key)
        else:
            base = obs

        arr = np.asarray(base) if base is not None else None
        if arr is None or arr.size == 0:
            return None

        if arr.ndim == 3:
            frame = arr[0]
        elif arr.ndim == 2:
            frame = arr
        else:
            return None

        return frame.astype(np.uint8, copy=False), spec.kind_norm()

    def _agent_surface(self, *, frame: tuple[np.ndarray, str], game: SnakeEngine):
        raw, kind = frame
        if kind == "pixel":
            return gray255_to_surface(raw, pixel_size=self.pixel_size)

        # categorical
        if self.agent_view_vocab is None:
            pixels = categorical_frame_to_pixels(
                raw,
                tile_size=int(game.tile_size),
            )
        else:
            pixels = _categorical_to_gray_pixels(
                raw,
                num_classes=int(self.agent_view_vocab.num_classes),
                tile_size=int(game.tile_size),
            )
        return gray255_to_surface(pixels, pixel_size=self.pixel_size)


def _categorical_to_gray_pixels(
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
