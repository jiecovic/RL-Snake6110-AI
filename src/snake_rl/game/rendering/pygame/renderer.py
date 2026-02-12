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

_BG = (0, 0, 0)
_PANEL_BG = (6, 6, 6)
_PANEL_BORDER = (24, 24, 24)
_HUD_BG = _PANEL_BG
_TEXT = (120, 255, 140)
_MUTED = (70, 140, 90)
_ACCENT = (0, 255, 90)


class PygameRenderer:
    def __init__(
        self,
        *,
        pixel_size: int = 10,
        agent_view_spec: ObservationSpec | None = None,
        agent_view_vocab: TileVocab | None = None,
        hud_mode: str = "all",
        hud_features: dict[str, Any] | None = None,
        hud_info: dict[str, str] | None = None,
    ):
        self.pixel_size = int(pixel_size)
        self.agent_view_spec = agent_view_spec
        self.agent_view_vocab = agent_view_vocab
        self.hud_mode = str(hud_mode)
        self.hud_features = dict(hud_features or {})
        self.hud_info = dict(hud_info or {})

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
        steps = int(game.episode_steps)
        paused = bool(ctx.paused)
        running = bool(game.running)

        status_pairs = [
            ("Score", f"{int(game.score)}"),
            ("Len", f"{int(game.snake_len)}"),
            ("Steps", f"{steps}"),
            ("State", "running" if running else "stopped"),
        ]
        perf_pairs = [
            ("Render", f"{fps:.1f} fps"),
            ("Sim", f"{sim:.1f} fps"),
            ("Target", f"{int(ctx.target_sim_hz)}"),
            ("Paused", "yes" if paused else "no"),
        ]
        feature_pairs = self._feature_items(game=game, include_snake_progress=True)
        info_pairs = self._info_pairs()

        controls = "[ ] speed  P pause  N step  R reset  A/D or Left/Right turn  Esc/Q quit"

        boxes = self._hud_layout(ctx)
        col_gap = int(ctx.layout.hud_col_gap)
        no_inner_borders = col_gap <= 0
        full_border = {"top", "bottom", "left", "right"}
        self._draw_hud_box(
            ctx=ctx,
            box=boxes["status"],
            title="STATUS",
            pairs=status_pairs,
            border_mask=full_border if not no_inner_borders else {"top", "bottom", "left"},
        )
        self._draw_hud_box(
            ctx=ctx,
            box=boxes["perf"],
            title="PERF",
            pairs=perf_pairs,
            border_mask=full_border if not no_inner_borders else {"top", "bottom"},
        )
        self._draw_hud_box(
            ctx=ctx,
            box=boxes["info"],
            title="INFO",
            pairs=info_pairs,
            empty_text="n/a",
            border_mask=full_border if not no_inner_borders else {"top", "bottom"},
        )
        self._draw_hud_box(
            ctx=ctx,
            box=boxes["features"],
            title="GLOBAL FEATURES",
            pairs=feature_pairs,
            empty_text="none",
            border_mask=full_border if not no_inner_borders else {"top", "bottom", "right"},
        )
        self._draw_hud_controls(ctx=ctx, text=controls)

    def _feature_items(
        self, *, game: SnakeEngine, include_snake_progress: bool = True
    ) -> list[tuple[str, str]]:
        flags = self._hud_feature_flags()
        items: list[tuple[str, str]] = []

        if flags.get("direction"):
            d = game.direction
            if d is None:
                dir_label = "-"
            else:
                dir_label = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}.get(int(d), "?")
            items.append(("Dir", dir_label))

        if include_snake_progress and flags.get("snake_progress"):
            items.append(("Snake", f"{float(game.snake_progress) * 100.0:.1f}%"))

        if flags.get("time_since_food"):
            t = float(game.time_since_food_norm(int(game.max_playable_tiles))) * 100.0
            items.append(("SinceFood", f"{t:.1f}%"))

        if flags.get("closest_food"):
            metric = self._closest_food_metric(flags.get("closest_food"))
            dx, dy, dist = game.get_closest_food_norm(metric)
            items.append(("Food", f"dx {dx:+.2f} dy {dy:+.2f} d {dist:.2f}"))

        if (
            flags.get("collision_ahead")
            or flags.get("collision_left")
            or flags.get("collision_right")
        ):
            ahead, left, right = game.collision_flags()
            items.append(("Coll", f"A {int(ahead)} L {int(left)} R {int(right)}"))

        return items

    def _closest_food_metric(self, value: Any) -> str:
        if isinstance(value, dict):
            metric = value.get("metric")
            if metric is not None:
                return str(metric)
        if isinstance(value, str):
            return str(value)
        return "manhattan"

    def _info_pairs(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        mode = str(self.hud_info.get("mode", "")).strip()
        if mode:
            items.append(("Mode", mode))
        run = str(self.hud_info.get("run", "")).strip()
        if run:
            items.append(("Run", run))
        which = str(self.hud_info.get("which", "")).strip()
        if which:
            items.append(("Which", which))
        seed = self.hud_info.get("seed")
        if seed is not None and str(seed).strip():
            items.append(("Seed", str(seed)))
        obs = self.hud_info.get("obs")
        if obs is None and self.agent_view_spec is not None:
            kind = self.agent_view_spec.kind_norm()
            view = self.agent_view_spec.view_norm()
            obs = f"{kind}/{view}"
        if obs:
            items.append(("Obs", str(obs)))
        action = str(self.hud_info.get("action", "")).strip()
        if action:
            items.append(("Action", action))
        return items

    def _draw_hud_box(
        self,
        *,
        ctx: PygameRenderContext,
        box: Rect,
        title: str,
        pairs: list[tuple[str, str]],
        empty_text: str | None = None,
        border_mask: set[str] | None = None,
    ) -> None:
        ctx.screen.fill(_PANEL_BG, box.to_tuple())
        if border_mask is None:
            ctx.screen.fill(_PANEL_BORDER, box.to_tuple(), 1)
        else:
            x, y, w, h = box.to_tuple()
            if "top" in border_mask:
                ctx.screen.fill(_PANEL_BORDER, (x, y, w, 1))
            if "bottom" in border_mask:
                ctx.screen.fill(_PANEL_BORDER, (x, y + h - 1, w, 1))
            if "left" in border_mask:
                ctx.screen.fill(_PANEL_BORDER, (x, y, 1, h))
            if "right" in border_mask:
                ctx.screen.fill(_PANEL_BORDER, (x + w - 1, y, 1, h))

        tx = int(box.x + 8)
        ty = int(box.y + 4)
        title_surf = ctx.label_font.render(str(title), True, _ACCENT)
        ctx.screen.blit(title_surf, (tx, ty))

        x = int(box.x + 8)
        y = int(box.y + 24)
        dy = ctx.label_font.get_height() + 4

        lines = self._format_pairs(pairs, empty_text=empty_text, key_pad=0)
        for i, line in enumerate(lines):
            ctx.screen.blit(ctx.label_font.render(line, True, _TEXT), (x, y + i * dy))

    def _draw_hud_controls(self, *, ctx: PygameRenderContext, text: str) -> None:
        hud = ctx.hud_panel
        x = int(hud.x + ctx.layout.hud_padding_x)
        y = int(hud.y + hud.h - ctx.layout.hud_padding_y - ctx.label_font.get_height())
        ctx.screen.blit(ctx.label_font.render(text, True, _MUTED), (x, y))

    def _hud_layout(self, ctx: PygameRenderContext) -> dict[str, Rect]:
        hud = ctx.hud_panel
        pad_x = int(ctx.layout.hud_padding_x)
        pad_y = int(ctx.layout.hud_padding_y)
        col_gap = int(ctx.layout.hud_col_gap)
        control_h = ctx.label_font.get_height() + 6

        inner_w = hud.w - 2 * pad_x - 3 * col_gap
        inner_h = hud.h - 2 * pad_y - control_h - int(ctx.layout.panel_gap)

        w_status = int(ctx.layout.hud_col_w_status)
        w_perf = int(ctx.layout.hud_col_w_perf)
        w_info = int(ctx.layout.hud_col_w_info)
        min_w = max(80, int(ctx.layout.hud_col_min))
        min_last = max(120, int(ctx.layout.hud_col_last_min))

        cols = [w_status, w_perf, w_info]
        last = inner_w - sum(cols)
        if last < min_last:
            deficit = min_last - last
            for i in (2, 1, 0):
                if deficit <= 0:
                    break
                reducible = max(0, cols[i] - min_w)
                take = min(deficit, reducible)
                cols[i] -= take
                deficit -= take
            last = inner_w - sum(cols)
        if last < min_last:
            if inner_w < 4 * min_w:
                w = inner_w // 4
                cols = [w, w, w]
                last = inner_w - 3 * w
            else:
                cols = [min_w, min_w, min_w]
                last = inner_w - 3 * min_w
        cols.append(last)

        x0 = int(hud.x + pad_x)
        top_y = int(hud.y + pad_y)
        col_x = [x0]
        for w in cols[:-1]:
            col_x.append(col_x[-1] + int(w) + col_gap)

        status = Rect(x=col_x[0], y=top_y, w=int(cols[0]), h=inner_h)
        perf = Rect(x=col_x[1], y=top_y, w=int(cols[1]), h=inner_h)
        info = Rect(x=col_x[2], y=top_y, w=int(cols[2]), h=inner_h)
        features = Rect(x=col_x[3], y=top_y, w=int(cols[3]), h=inner_h)

        return {
            "status": status,
            "perf": perf,
            "info": info,
            "features": features,
        }

    @staticmethod
    def _format_pairs(
        pairs: list[tuple[str, str]],
        *,
        empty_text: str | None = None,
        key_pad: int = 0,
    ) -> list[str]:
        if not pairs:
            return [str(empty_text)] if empty_text is not None else []
        key_w = max(len(k) for k, _v in pairs) + max(0, int(key_pad))
        return [f"{k:<{key_w}} {v}" for k, v in pairs]

    @staticmethod
    def _format_section(
        label: str,
        pairs: list[tuple[str, str]],
        *,
        label_w: int,
        per_line: int = 2,
        empty_text: str | None = None,
    ) -> list[str]:
        if not pairs:
            if empty_text is None:
                return []
            return [f"{label:<{label_w}} {empty_text}"]

        lines: list[str] = []
        for i in range(0, len(pairs), per_line):
            chunk = pairs[i : i + per_line]
            parts = [f"{k} {v}" for k, v in chunk]
            tag = label if i == 0 else ""
            lines.append(f"{tag:<{label_w}} " + " | ".join(parts))
        return lines

    def _hud_feature_flags(self) -> dict[str, Any]:
        mode = str(self.hud_mode).strip().lower()
        if mode == "selected":
            return {k: v for k, v in self.hud_features.items() if v}
        return {
            "direction": True,
            "snake_progress": True,
            "time_since_food": True,
            "closest_food": True,
            "collision_ahead": True,
            "collision_left": True,
            "collision_right": True,
        }

    def _agent_label(self) -> str:
        if self.agent_view_spec is None:
            return ""
        kind = self.agent_view_spec.kind_norm()
        view = self.agent_view_spec.view_norm()
        vocab = self.agent_view_vocab.name if self.agent_view_vocab is not None else None
        if vocab:
            return f"{kind}/{view} - {vocab}"
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
