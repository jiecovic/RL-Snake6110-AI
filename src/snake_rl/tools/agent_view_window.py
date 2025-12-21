# src/snake_rl/tools/agent_view_window.py
from __future__ import annotations

import argparse
import pickle
import sys
from typing import Optional, Tuple

import numpy as np
import pygame

from snake_rl.game.rendering.pygame.surf import gray255_to_surface


def _read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return b""
        buf += chunk
    return buf


def _read_msg() -> Optional[dict]:
    hdr = _read_exact(4)
    if not hdr:
        return None
    size = int.from_bytes(hdr, "big", signed=False)
    payload = _read_exact(size)
    if not payload:
        return None
    return pickle.loads(payload)


def _as_2d(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise TypeError(f"expected 2D frame, got shape={arr.shape}")
    return arr


def _as_u8_2d(frame: np.ndarray) -> np.ndarray:
    arr = _as_2d(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _choose_pixel_size(*, h: int, w: int, max_size: int) -> int:
    m = max(1, int(max(h, w)))
    return max(1, int(max_size) // m)


def _ids_to_gray255(ids2d: np.ndarray, *, num_classes: int) -> np.ndarray:
    # ids2d can be int/uint; normalize to [0,255] for background shading.
    arr = np.asarray(ids2d)
    if arr.dtype.kind not in {"u", "i"}:
        arr = arr.astype(np.int64, copy=False)

    k = int(max(1, num_classes))
    if k <= 1:
        return np.zeros(arr.shape, dtype=np.uint8)

    # Clip into valid range then scale.
    arr = np.clip(arr, 0, k - 1)
    gray = (arr.astype(np.float32) / float(k - 1)) * 255.0
    return gray.astype(np.uint8, copy=False)


def _render_ids_overlay(
        *,
        screen: pygame.Surface,
        ids2d: np.ndarray,
        ps: int,
        font: pygame.font.Font,
        font_color=(255, 255, 255),
        outline_color=(0, 0, 0),
) -> None:
    # ids2d is [H,W]
    h, w = int(ids2d.shape[0]), int(ids2d.shape[1])

    # Ensure ints for rendering.
    if ids2d.dtype.kind not in {"u", "i"}:
        ids = ids2d.astype(np.int64, copy=False)
    else:
        ids = ids2d

    # Small outline makes numbers readable on both dark/light backgrounds.
    for y in range(h):
        for x in range(w):
            v = int(ids[y, x])
            s = str(v)

            text = font.render(s, True, font_color)
            outline = font.render(s, True, outline_color)

            cx = x * ps + ps // 2
            cy = y * ps + ps // 2
            rect = text.get_rect(center=(cx, cy))

            # 1px outline in 4 directions
            orect = rect.copy()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                orect.center = (cx + dx, cy + dy)
                screen.blit(outline, orect)

            screen.blit(text, rect)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent-view window helper (spawned by snake-watch).")
    ap.add_argument("--caption", type=str, default="Snake (agent view)")
    ap.add_argument("--max-size", type=int, default=480, help="Max window side in pixels (auto pixel_size).")
    ap.add_argument("--fps", type=int, default=0, help="If >0, cap render FPS. 0 => render on every update.")
    ap.add_argument("--pixel-size", type=int, default=0, help="If >0, force this pixel_size (match main window).")
    args = ap.parse_args()

    caption = str(args.caption)
    max_size = max(64, int(args.max_size))
    cap_fps = max(0, int(args.fps))
    forced_ps = max(0, int(args.pixel_size))

    pygame.init()
    clock = pygame.time.Clock()

    screen: Optional[pygame.Surface] = None
    last_size: Tuple[int, int] = (0, 0)

    # Font is created lazily once we know pixel_size.
    font: Optional[pygame.font.Font] = None
    last_font_ps: int = 0

    try:
        pygame.display.set_caption(caption)

        while True:
            # Keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            msg = _read_msg()
            if msg is None:
                return

            if msg.get("type") == "close":
                return
            if msg.get("type") != "frame":
                continue

            frame = msg.get("frame", None)
            if frame is None:
                continue

            mode = str(msg.get("mode", "gray255"))

            # --- decode frame ---
            try:
                f_any = _as_2d(frame)
            except Exception:
                continue

            h, w = int(f_any.shape[0]), int(f_any.shape[1])
            ps = forced_ps if forced_ps > 0 else _choose_pixel_size(h=h, w=w, max_size=max_size)
            target = (max(1, w * ps), max(1, h * ps))

            if screen is None or target != last_size:
                screen = pygame.display.set_mode(target)
                pygame.display.set_caption(caption)
                last_size = target

            assert screen is not None

            # --- render ---
            if mode == "ids":
                num_classes_v = msg.get("num_classes", None)
                if num_classes_v is None:
                    # Fallback: render as gray255 (best-effort)
                    f = _as_u8_2d(f_any)
                    surf = gray255_to_surface(f, pixel_size=ps)
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                else:
                    try:
                        num_classes = int(num_classes_v)
                    except Exception:
                        num_classes = 0

                    bg = _ids_to_gray255(f_any, num_classes=num_classes if num_classes > 0 else 1)
                    surf = gray255_to_surface(bg, pixel_size=ps)
                    screen.blit(surf, (0, 0))

                    # Font size relative to cell size.
                    # Keep it readable but not overflowing.
                    desired = max(8, int(ps * 0.55))
                    if font is None or last_font_ps != desired:
                        # SysFont is portable; monospace helps alignment.
                        font = pygame.font.SysFont("consolas", desired)
                        last_font_ps = desired

                    _render_ids_overlay(
                        screen=screen,
                        ids2d=f_any,
                        ps=ps,
                        font=font,
                    )
                    pygame.display.flip()

            else:
                # Single canonical rendering path: frame is uint8 [0,255]
                try:
                    f = _as_u8_2d(f_any)
                except Exception:
                    continue
                surf = gray255_to_surface(f, pixel_size=ps)
                screen.blit(surf, (0, 0))
                pygame.display.flip()

            if cap_fps > 0:
                clock.tick(cap_fps)

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
