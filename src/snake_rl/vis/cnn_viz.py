# src/snake_rl/vis/cnn_viz.py
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np


def _normalize(img: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(img))
    vmax = float(np.nanmax(img))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def _tile_images(images: list[np.ndarray], *, ncols: int) -> np.ndarray:
    if not images:
        return np.zeros((1, 1), dtype=np.float32)
    h, w = images[0].shape[:2]
    ncols = max(1, int(ncols))
    ncols = min(ncols, len(images))
    nrows = int(ceil(len(images) / float(ncols)))
    grid = np.zeros((nrows * h, ncols * w), dtype=np.float32)
    for idx, img in enumerate(images):
        r = idx // ncols
        c = idx % ncols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return grid


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    k = int(max(1, min(int(k), scores.shape[0])))
    return np.argsort(scores)[::-1][:k]


@dataclass
class CnnVizConfig:
    k: int = 16
    update_every: int = 1
    ncols: int = 8


class CnnVisualizer:
    """
    Lightweight CNN visualization for watch mode.

    Shows:
      - first conv activations
      - last conv activations (pre-flatten)
      - first conv kernels
    """

    def __init__(self, *, model: Any, config: CnnVizConfig, logger: Any | None = None) -> None:
        self._logger = logger
        self._config = config
        self._step = 0

        self._model: Any | None = None
        self._first_conv = None
        self._last_conv = None
        self._hook_handles: list[Any] = []

        self._first_act: np.ndarray | None = None
        self._last_act: np.ndarray | None = None
        self._kernel_grid: np.ndarray | None = None

        self._plt = None
        self._fig = None
        self._axes = None
        self._im_first = None
        self._im_last = None
        self._im_kernel = None

        self._init_mpl()
        self.set_model(model)

    def _init_mpl(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required for CNN visualization") from exc

        self._plt = plt
        self._plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("CNN Visualizer")
        for ax, title in zip(
            axes,
            ("conv1 activations", "conv_last activations", "conv1 kernels"),
            strict=False,
        ):
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
        self._fig = fig
        self._axes = axes
        self._im_first = axes[0].imshow(
            np.zeros((1, 1)),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        self._im_last = axes[1].imshow(
            np.zeros((1, 1)),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        self._im_kernel = axes[2].imshow(
            np.zeros((1, 1)),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        with suppress(Exception):
            manager = getattr(fig.canvas, "manager", None)
            if manager is not None:
                manager.set_window_title("Snake CNN Visualizer")
        self._plt.show(block=False)

    def _log(self, msg: str) -> None:
        if self._logger is not None:
            self._logger.info(msg)

    def _detach_hooks(self) -> None:
        for h in self._hook_handles:
            with suppress(Exception):
                h.remove()
        self._hook_handles.clear()

    def _find_convs(self, model: Any) -> list[Any]:
        try:
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("torch is required for CNN visualization") from exc

        policy = getattr(model, "policy", None)
        if policy is None:
            raise ValueError("model has no policy")
        feat = getattr(policy, "features_extractor", None)
        if feat is None:
            raise ValueError("policy has no features_extractor")

        convs = [m for m in feat.modules() if isinstance(m, nn.Conv2d)]
        return convs

    def _hook_first(self, _module, _inputs, output) -> None:
        out = output[0] if isinstance(output, (tuple, list)) and output else output
        try:
            import torch
        except Exception:
            self._first_act = None
            return
        if torch.is_tensor(out):
            self._first_act = out.detach().float().cpu().numpy()
        else:
            self._first_act = None

    def _hook_last(self, _module, _inputs, output) -> None:
        out = output[0] if isinstance(output, (tuple, list)) and output else output
        try:
            import torch
        except Exception:
            self._last_act = None
            return
        if torch.is_tensor(out):
            self._last_act = out.detach().float().cpu().numpy()
        else:
            self._last_act = None

    def set_model(self, model: Any) -> None:
        self._model = model
        self._detach_hooks()
        convs = self._find_convs(model)
        if not convs:
            raise ValueError("no Conv2d layers found in feature extractor")

        self._first_conv = convs[0]
        self._last_conv = convs[-1]

        self._hook_handles.append(self._first_conv.register_forward_hook(self._hook_first))
        if self._last_conv is not self._first_conv:
            self._hook_handles.append(self._last_conv.register_forward_hook(self._hook_last))
        else:
            self._hook_handles.append(self._first_conv.register_forward_hook(self._hook_last))

        self._kernel_grid = self._make_kernel_grid()
        if self._im_kernel is not None and self._kernel_grid is not None:
            self._im_kernel.set_data(self._kernel_grid)

    def _make_kernel_grid(self) -> np.ndarray | None:
        if self._first_conv is None:
            return None
        try:
            weight = self._first_conv.weight.detach().float().cpu().numpy()
        except Exception:
            return None
        if weight.ndim != 4:
            return None
        # weight shape: (out_c, in_c, kH, kW)
        filt = weight.mean(axis=1) if weight.shape[1] > 1 else weight[:, 0, :, :]
        scores = np.mean(np.abs(filt), axis=(1, 2))
        idx = _topk_indices(scores, self._config.k)
        tiles = [_normalize(filt[i]) for i in idx]
        return _tile_images(tiles, ncols=self._config.ncols)

    def _make_act_grid(self, act: np.ndarray | None) -> np.ndarray | None:
        if act is None or act.ndim < 3:
            return None
        x = act[0] if act.ndim == 4 else act
        if x.ndim != 3:
            return None
        scores = np.mean(np.abs(x), axis=(1, 2))
        idx = _topk_indices(scores, self._config.k)
        tiles = [_normalize(x[i]) for i in idx]
        return _tile_images(tiles, ncols=self._config.ncols)

    def _capture(self, obs: Any) -> None:
        if self._model is None:
            return
        try:
            import torch
        except Exception:
            return
        try:
            obs_tensor, _ = self._model.policy.obs_to_tensor(obs)
        except Exception:
            return
        with torch.no_grad():
            _ = self._model.policy.extract_features(obs_tensor)

    def update(self, obs: Any) -> None:
        self._step += 1
        if self._step % max(1, int(self._config.update_every)) != 0:
            return
        if self._plt is None or self._fig is None:
            return
        self._capture(obs)

        first_grid = self._make_act_grid(self._first_act)
        last_grid = self._make_act_grid(self._last_act)

        if self._im_first is not None and first_grid is not None:
            self._im_first.set_data(first_grid)
            self._im_first.set_clim(0.0, 1.0)
        if self._im_last is not None and last_grid is not None:
            self._im_last.set_data(last_grid)
            self._im_last.set_clim(0.0, 1.0)
        if self._im_kernel is not None and self._kernel_grid is not None:
            self._im_kernel.set_data(self._kernel_grid)
            self._im_kernel.set_clim(0.0, 1.0)

        with suppress(Exception):
            self._fig.canvas.draw_idle()
            self._plt.pause(0.001)

    def close(self) -> None:
        self._detach_hooks()
        if self._plt is not None and self._fig is not None:
            with suppress(Exception):
                self._plt.close(self._fig)
