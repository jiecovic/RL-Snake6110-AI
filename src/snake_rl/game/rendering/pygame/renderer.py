# src/snake_rl/game/rendering/pygame/renderer.py
from __future__ import annotations

from typing import Any

import numpy as np

from snake_rl.envs.specs import ObservationSpec
from snake_rl.game.rendering.pygame.draw import (
    categorical_to_gray_pixels,
    draw_hud_box,
    draw_hud_controls,
    draw_panel,
)
from snake_rl.game.rendering.pygame.hud import (
    feature_flags_from_spec,
    feature_items,
    hud_feature_flags,
    info_pairs,
)
from snake_rl.game.rendering.pygame.layout import hud_layout
from snake_rl.game.rendering.pygame.surf import gray255_to_surface
from snake_rl.game.rendering.pygame.theme import BG, HUD_BG, PANEL_BORDER
from snake_rl.game.rendering.pygame.window import PygameRenderContext
from snake_rl.game.snake_engine import SnakeEngine
from snake_rl.utils.obs_render import categorical_frame_to_pixels


class PygameRenderer:
    def __init__(
        self,
        *,
        pixel_size: int = 10,
        agent_view_spec: ObservationSpec | None = None,
        agent_view_vocab_name: str | None = None,
        agent_view_vocab_num_classes: int | None = None,
        hud_mode: str = "all",
        hud_features: dict[str, Any] | None = None,
        hud_info: dict[str, str] | None = None,
    ):
        self.pixel_size = int(pixel_size)
        self.agent_view_spec = agent_view_spec
        if agent_view_spec is not None and agent_view_vocab_name is None:
            agent_view_vocab_name = agent_view_spec.tile_vocab_name()
        if (
            agent_view_spec is not None
            and agent_view_vocab_num_classes is None
            and agent_view_vocab_name is not None
        ):
            agent_view_vocab_num_classes = agent_view_spec.tile_vocab_num_classes()
        self.agent_view_vocab_name = agent_view_vocab_name
        self.agent_view_vocab_num_classes = agent_view_vocab_num_classes
        self.hud_mode = str(hud_mode)
        self.hud_features = dict(hud_features or {})
        self.hud_info = hud_info if hud_info is not None else {}

    def draw(self, *, game: SnakeEngine, ctx: PygameRenderContext) -> None:
        if game.pixel_buffer is None or game.pixel_buffer.size == 0:
            return

        ctx.screen.fill(BG)

        draw_panel(
            ctx=ctx,
            panel=ctx.world_panel,
            title="WORLD",
            subtitle=f"{int(game.width)}x{int(game.height)} tiles",
        )
        if ctx.agent_panel is not None:
            label = self._agent_label()
            draw_panel(ctx=ctx, panel=ctx.agent_panel, title="AGENT", subtitle=label)

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

    def _draw_hud(self, *, game: SnakeEngine, ctx: PygameRenderContext) -> None:
        hud = ctx.hud_panel
        ctx.screen.fill(HUD_BG, hud.to_tuple())
        ctx.screen.fill(PANEL_BORDER, hud.to_tuple(), 1)

        fps = float(ctx.clock.get_fps())
        sim = float(ctx.sim_fps)
        steps = int(game.episode_steps)
        paused = bool(ctx.paused)
        running = bool(game.running)
        wins = self.hud_info.get("wins")

        status_pairs = [
            ("Score", f"{int(game.score)}"),
            ("Len", f"{int(game.snake_len)}"),
            ("Steps", f"{steps}"),
        ]
        if wins is not None:
            status_pairs.append(("Wins", str(wins)))
        status_pairs.extend(
            [
                ("State", "running" if running else "stopped"),
            ]
        )
        perf_pairs = [
            ("Render", f"{fps:.1f} fps"),
            ("Sim", f"{sim:.1f} fps"),
            ("Target", f"{int(ctx.target_sim_hz)}"),
            ("Paused", "yes" if paused else "no"),
        ]
        flags = hud_feature_flags(hud_mode=self.hud_mode, hud_features=self.hud_features)
        if not flags and self.hud_mode.strip().lower() == "selected" and self.agent_view_spec:
            flags = feature_flags_from_spec(self.agent_view_spec)
        feature_pairs = feature_items(game=game, flags=flags, include_snake_progress=True)
        info_items = info_pairs(hud_info=self.hud_info, agent_view_spec=self.agent_view_spec)

        controls = "[ ] speed  P pause  N step  R reset  A/D or Left/Right turn  Esc/Q quit"

        boxes = hud_layout(ctx)
        col_gap = int(ctx.layout.hud_col_gap)
        no_inner_borders = col_gap <= 0
        full_border = {"top", "bottom", "left", "right"}
        draw_hud_box(
            ctx=ctx,
            box=boxes["status"],
            title="STATUS",
            pairs=status_pairs,
            border_mask=full_border if not no_inner_borders else {"top", "bottom", "left"},
        )
        draw_hud_box(
            ctx=ctx,
            box=boxes["perf"],
            title="PERF",
            pairs=perf_pairs,
            border_mask=full_border if not no_inner_borders else {"top", "bottom"},
        )
        draw_hud_box(
            ctx=ctx,
            box=boxes["info"],
            title="INFO",
            pairs=info_items,
            empty_text="n/a",
            border_mask=full_border if not no_inner_borders else {"top", "bottom"},
        )
        draw_hud_box(
            ctx=ctx,
            box=boxes["features"],
            title="GLOBAL FEATURES",
            pairs=feature_pairs,
            empty_text="none",
            border_mask=full_border if not no_inner_borders else {"top", "bottom", "right"},
        )
        draw_hud_controls(ctx=ctx, text=controls)

    def _agent_label(self) -> str:
        if self.agent_view_spec is None:
            return ""
        kind = self.agent_view_spec.kind_norm()
        view = self.agent_view_spec.view_norm()
        vocab = self.agent_view_vocab_name
        if vocab:
            return f"{kind}/{view} - {vocab}"
        return f"{kind}/{view}"

    def _build_agent_frame(self, *, game: SnakeEngine) -> tuple[np.ndarray, str] | None:
        if self.agent_view_spec is None:
            return None
        spec = self.agent_view_spec
        obs = spec.observe(
            game=game,
            initial_snake_length=int(game.snake_len),
            max_playable_tiles=int(game.max_playable_tiles),
            max_steps=int(game.max_playable_tiles),
            frame_stack_n=int(game.frame_stack_n),
        )
        base: Any
        if isinstance(obs, dict):
            key = spec.frame_stack_key() or (
                "pixel" if spec.kind_norm() == "pixel" else "categorical"
            )
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
        if self.agent_view_vocab_name is None:
            pixels = categorical_frame_to_pixels(
                raw,
                tile_size=int(game.tile_size),
            )
        else:
            if self.agent_view_vocab_num_classes is None:
                raise RuntimeError("agent_view_vocab_num_classes is required for vocab rendering")
            pixels = categorical_to_gray_pixels(
                raw,
                num_classes=int(self.agent_view_vocab_num_classes),
                tile_size=int(game.tile_size),
            )
        return gray255_to_surface(pixels, pixel_size=self.pixel_size)
