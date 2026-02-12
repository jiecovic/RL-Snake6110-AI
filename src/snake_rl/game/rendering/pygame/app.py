# src/snake_rl/game/rendering/pygame/app.py
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snake_rl import _core as core
from snake_rl.game.rendering.pygame.renderer import PygameRenderer
from snake_rl.game.rendering.pygame.window import (
    LayoutConfig,
    PygameRenderContext,
    create_pygame_context,
)
from snake_rl.game.snake_engine import SnakeEngine

if TYPE_CHECKING:
    from snake_rl.envs.specs import ObservationSpec

try:
    import pygame as _pygame
except Exception:  # pragma: no cover
    _pygame = None

pygame: Any = _pygame

StepFn = Callable[[], None]


@dataclass(slots=True)
class AppConfig:
    fps: int
    sim_hz: int | None = None
    pixel_size: int = 10
    caption: str = "Snake"
    reset_on_done: bool = True
    max_steps_per_frame: int = 5
    # Human input (optional)
    enable_human_input: bool = False
    turn_keys: tuple[int, int] | None = None  # left, right
    # Agent view (optional)
    agent_view_spec: ObservationSpec | None = None
    agent_view_vocab_name: str | None = None
    agent_view_vocab_num_classes: int | None = None
    # HUD
    hud_mode: str = "all"  # "all" | "selected"
    hud_features: dict[str, Any] | None = None
    hud_info: dict[str, str] | None = None
    layout: LayoutConfig = field(default_factory=LayoutConfig)


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

    Timing:
      - Render is capped at cfg.fps (<= 0 means uncapped).
      - Simulation is stepped at cfg.sim_hz (defaults to fps).
      - Steps are decoupled from rendering via a fixed-timestep accumulator.
    """
    if pygame is None:  # pragma: no cover
        raise RuntimeError("pygame is not installed")

    pygame.init()
    try:
        agent_view_grid = _compute_agent_view_grid(game=game, spec=cfg.agent_view_spec)
        ctx: PygameRenderContext = create_pygame_context(
            game=game,
            pixel_size=cfg.pixel_size,
            caption=cfg.caption,
            agent_view_grid=agent_view_grid,
            layout=cfg.layout,
        )
        renderer = PygameRenderer(
            pixel_size=cfg.pixel_size,
            agent_view_spec=cfg.agent_view_spec,
            agent_view_vocab_name=cfg.agent_view_vocab_name,
            agent_view_vocab_num_classes=cfg.agent_view_vocab_num_classes,
            hud_mode=str(cfg.hud_mode),
            hud_features=cfg.hud_features,
            hud_info=cfg.hud_info,
        )

        paused = False
        queued_turn: int | None = None  # buffered human input

        if cfg.turn_keys is None:
            left_keys = {pygame.K_a, pygame.K_LEFT}
            right_keys = {pygame.K_d, pygame.K_RIGHT}
        else:
            left_key, right_key = cfg.turn_keys
            left_keys = {left_key}
            right_keys = {right_key}

        sim_hz = int(cfg.sim_hz) if cfg.sim_hz is not None else int(cfg.fps)
        sim_hz = max(1, sim_hz)
        ctx.target_sim_hz = int(sim_hz)
        step_dt = 1.0 / float(sim_hz)

        last_time = time.perf_counter()
        accumulator = 0.0
        sim_tick_t0 = last_time
        sim_steps = 0
        total_steps = 0

        speed_down = False
        speed_up = False
        step_once = False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # NOTE: In step_fn mode, resetting only the game can desync env state.
                        # We keep this for legacy/human mode. For watch, don't press R.
                        game.reset()
                        paused = False
                        queued_turn = None
                    elif event.key == pygame.K_n:
                        step_once = True
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_down = True
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_up = True
                    elif cfg.enable_human_input:
                        # Buffer exactly one upcoming turn (latest wins)
                        if event.key in left_keys:
                            queued_turn = 1
                        elif event.key in right_keys:
                            queued_turn = 2
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFTBRACKET:
                        speed_down = False
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_up = False

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            if paused:
                accumulator = 0.0
                if step_once:
                    if step_fn is not None:
                        step_fn()
                    else:
                        if game.running:
                            if cfg.enable_human_input:
                                rel = queued_turn if queued_turn is not None else 0
                                queued_turn = None
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
                    sim_steps += 1
                    total_steps += 1
                    step_once = False
            else:
                if speed_down or speed_up:
                    delta = 5 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
                    if speed_down:
                        sim_hz = max(1, int(sim_hz) - delta)
                    if speed_up:
                        sim_hz = min(1000, int(sim_hz) + delta)
                    step_dt = 1.0 / float(sim_hz)
                    ctx.target_sim_hz = int(sim_hz)

                accumulator += dt

                steps = 0
                while accumulator >= step_dt and steps < int(cfg.max_steps_per_frame):
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

                    accumulator -= step_dt
                    steps += 1
                    sim_steps += 1
                    total_steps += 1

                # Prevent spiral of death when rendering stalls.
                if steps >= int(cfg.max_steps_per_frame):
                    accumulator = 0.0

            now = time.perf_counter()
            elapsed = now - sim_tick_t0
            if elapsed >= 1.0:
                ctx.sim_fps = float(sim_steps) / elapsed
                sim_steps = 0
                sim_tick_t0 = now
            ctx.sim_steps = int(total_steps)
            ctx.paused = bool(paused)

            renderer.draw(game=game, ctx=ctx)
            pygame.display.flip()
            if int(cfg.fps) > 0:
                ctx.clock.tick(cfg.fps)
            else:
                ctx.clock.tick(0)
    finally:
        pygame.quit()


def _compute_agent_view_grid(
    *,
    game: SnakeEngine,
    spec: ObservationSpec | None,
) -> tuple[int, int] | None:
    if spec is None:
        return None
    kind = spec.kind_norm()
    view = spec.view_norm()
    tile_size = int(game.tile_size)

    if view != "head":
        return None

    ry, rx = spec._view_radius()
    view_h = (2 * int(ry) + 1) * tile_size
    view_w = (2 * int(rx) + 1) * tile_size
    if kind not in {"pixel", "categorical"}:
        return None
    return (view_w, view_h)
