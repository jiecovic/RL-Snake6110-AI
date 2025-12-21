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


def _as_u8_2d(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise TypeError(f"expected 2D frame, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _choose_pixel_size(*, h: int, w: int, max_size: int) -> int:
    m = max(1, int(max(h, w)))
    return max(1, int(max_size) // m)


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

            try:
                f = _as_u8_2d(frame)
            except Exception:
                continue

            h, w = int(f.shape[0]), int(f.shape[1])
            ps = forced_ps if forced_ps > 0 else _choose_pixel_size(h=h, w=w, max_size=max_size)
            target = (max(1, w * ps), max(1, h * ps))

            if screen is None or target != last_size:
                screen = pygame.display.set_mode(target)
                pygame.display.set_caption(caption)
                last_size = target

            assert screen is not None

            # Single canonical rendering path: frame is uint8 [0,255]
            surf = gray255_to_surface(f, pixel_size=ps)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if cap_fps > 0:
                clock.tick(cap_fps)

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
