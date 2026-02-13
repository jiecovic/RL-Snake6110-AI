# src/snake_rl/models/stems/px_cnn.py
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from snake_rl.models.stems.base import TokenStem, TokenStemOutput


def _act(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n in {"relu", "r"}:
        return nn.ReLU()
    if n in {"gelu", "g"}:
        return nn.GELU()
    if n in {"silu", "swish"}:
        return nn.SiLU()
    if n in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Unknown activation {name!r}")


def _nature_layers() -> list[dict[str, Any]]:
    return [
        {"out": 32, "k": 8, "s": 4, "p": 0, "act": "relu"},
        {"out": 64, "k": 4, "s": 2, "p": 0, "act": "relu"},
        {"out": 64, "k": 3, "s": 1, "p": 0, "act": "relu"},
    ]


def _maybe_pool(layer: dict[str, Any]) -> nn.Module | None:
    pool = layer.get("pool")
    if pool is None or pool is False:
        return None
    if pool is True:
        return nn.AvgPool2d(kernel_size=2, stride=2)
    if isinstance(pool, str):
        name = pool.strip().lower()
        if name in {"avg", "average"}:
            return nn.AvgPool2d(kernel_size=2, stride=2)
        if name in {"max"}:
            return nn.MaxPool2d(kernel_size=2, stride=2)
        raise ValueError(f"Unknown pool type {pool!r}")
    if isinstance(pool, dict):
        name = str(pool.get("type", "avg")).strip().lower()
        k = int(pool.get("k", 2))
        s = int(pool.get("s", k))
        p = int(pool.get("p", 0))
        if name in {"avg", "average"}:
            return nn.AvgPool2d(kernel_size=k, stride=s, padding=p)
        if name in {"max"}:
            return nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        raise ValueError(f"Unknown pool type {name!r}")
    raise ValueError(f"Invalid pool spec {pool!r}")


class PxCnnStem(TokenStem):
    def __init__(
        self,
        *,
        in_channels: int,
        d_model: int,
        layers: list[dict[str, Any]] | None = None,
        preset: str | None = None,
        frame_mode: str = "stacked",
        frame_base_channels: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.frame_mode = str(frame_mode)
        if self.frame_mode not in {"stacked", "per_frame"}:
            raise ValueError(f"frame_mode must be 'stacked' or 'per_frame', got {frame_mode!r}")
        self.frame_base_channels = int(frame_base_channels)
        if self.frame_base_channels <= 0:
            raise ValueError("frame_base_channels must be >= 1")

        if preset:
            p = str(preset).strip().lower()
            if p != "nature":
                raise ValueError(f"Unknown preset {preset!r}")
            layers = _nature_layers()
        if not layers:
            raise ValueError("PxCnnStem requires non-empty layers or preset='nature'")

        modules: list[nn.Module] = []
        in_ch = int(in_channels)
        last_out = in_ch
        for layer in layers:
            out_raw = layer.get("out")
            if out_raw is None:
                raise ValueError("PxCnnStem layer missing required key 'out'")
            out_ch = int(out_raw)
            k = int(layer.get("k", 3))
            s = int(layer.get("s", 1))
            p = int(layer.get("p", 0))
            act = layer.get("act", "relu")
            modules.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            modules.append(_act(act))
            pool = _maybe_pool(layer)
            if pool is not None:
                modules.append(pool)
            in_ch = out_ch
            last_out = out_ch

        self.stem = nn.Sequential(*modules)
        self.proj: nn.Module
        if int(last_out) != int(self.d_model):
            self.proj = nn.Conv2d(int(last_out), int(self.d_model), kernel_size=1, stride=1)
        else:
            self.proj = nn.Identity()

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.proj(y)
        return y

    def forward(self, obs: torch.Tensor) -> TokenStemOutput:
        if obs.ndim == 3:
            obs = obs.unsqueeze(1)
        if obs.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got {tuple(obs.shape)}")

        obs = obs.float()
        b, c, h, w = obs.shape
        if self.frame_mode == "per_frame":
            if c % self.frame_base_channels != 0:
                raise ValueError(
                    f"channels {c} not divisible by frame_base_channels {self.frame_base_channels}"
                )
            frames = c // self.frame_base_channels
            x = obs.reshape(b * frames, self.frame_base_channels, h, w)
            y = self._forward_cnn(x)
            _, d, h2, w2 = y.shape
            tokens = y.flatten(2).transpose(1, 2)  # [B*F, T, D]
            tokens = tokens.reshape(b, frames * h2 * w2, d)
            return TokenStemOutput(
                tokens=tokens,
                mask=None,
                grid=(int(h2), int(w2)),
                frame_count=int(frames),
                frame_fuse="concat",
            )

        y = self._forward_cnn(obs.float())
        _, d, h2, w2 = y.shape
        tokens = y.flatten(2).transpose(1, 2)
        return TokenStemOutput(
            tokens=tokens,
            mask=None,
            grid=(int(h2), int(w2)),
            frame_count=1,
            frame_fuse="sum",
        )
