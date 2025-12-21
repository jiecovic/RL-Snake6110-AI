# src/snake_rl/tools/agent_view_stream.py
from __future__ import annotations

import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _send_msg(proc: subprocess.Popen, msg: dict) -> None:
    if proc.stdin is None:
        return
    payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(len(payload).to_bytes(4, "big"))
    proc.stdin.write(payload)
    proc.stdin.flush()


@dataclass
class AgentViewStream:
    """
    Spawns a separate process to render frames (second window)
    and streams frames + metadata to it.

    Protocol (msg dict):
      type="frame"
        - frame: np.ndarray
        - mode: "gray255" | "ids"        (optional, default="gray255")
        - num_classes: int | None        (required for mode="ids")
    """

    proc: Optional[subprocess.Popen] = None
    last_send_t: float = 0.0

    def start(
            self,
            *,
            caption: str,
            max_size: int,
            fps: int,
            keep_stderr: bool = False,
            pixel_size: int = 0,  # 0 => auto
    ) -> None:
        cmd = [
            sys.executable,
            "-m",
            "snake_rl.tools.agent_view_window",
            "--caption",
            str(caption),
            "--max-size",
            str(int(max_size)),
            "--fps",
            str(int(fps)),
            "--pixel-size",
            str(int(pixel_size)),
        ]

        stderr = None if keep_stderr else subprocess.DEVNULL

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                _send_msg(self.proc, {"type": "close"})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None

    def send_frame(
            self,
            frame: np.ndarray,
            *,
            mode: str = "gray255",
            num_classes: Optional[int] = None,
            max_fps: int = 0,
    ) -> None:
        """
        Send a frame to the agent-view window.

        Parameters
        ----------
        frame:
            - gray255 mode: uint8 array [H,W] in [0,255]
            - ids mode:     uint8 / int array [H,W] with class IDs
        mode:
            "gray255" (default) or "ids"
        num_classes:
            Required for mode="ids"; ignored otherwise.
        max_fps:
            If >0, throttle sends to this FPS.
        """
        if not self.is_alive():
            return

        if max_fps > 0:
            now = time.time()
            if (now - self.last_send_t) < (1.0 / float(max_fps)):
                return
            self.last_send_t = now

        try:
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)

            msg = {
                "type": "frame",
                "frame": frame,
                "mode": str(mode),
            }

            if mode == "ids":
                if num_classes is None:
                    raise ValueError("send_frame(mode='ids') requires num_classes")
                msg["num_classes"] = int(num_classes)

            _send_msg(self.proc, msg)
        except Exception:
            # Never crash the main loop because of the debug window
            pass
