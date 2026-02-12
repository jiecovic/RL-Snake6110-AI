# src/snake_rl/models/utils/feature_tokens.py
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

DEFAULT_CATEGORICAL: dict[str, int] = {
    "direction": 4,
}


def normalize_feature_groups(groups: Iterable[Iterable[str]] | None) -> list[list[str]]:
    if groups is None:
        return []
    out: list[list[str]] = []
    for g in groups:
        items = [str(g)] if isinstance(g, str) else [str(x) for x in g]
        items = [s.strip() for s in items if str(s).strip()]
        if not items:
            raise ValueError("feature_tokens entries must be non-empty")
        out.append(items)
    return out


def _feature_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim == 2:
        return x
    raise ValueError(f"feature tensors must be shape (B,) or (B,K); got {tuple(x.shape)}")


def _as_cat_ids(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    t = _feature_tensor(x)
    if t.shape[1] == 1:
        t = t.squeeze(1).to(dtype=torch.long)
    elif num_classes > 0 and t.shape[1] == num_classes:
        # Accept one-hot or probability-like vectors.
        t = torch.argmax(t, dim=1).to(dtype=torch.long)
    else:
        raise ValueError(
            "categorical features must have shape (B,) or (B,1) "
            f"or one-hot (B,{num_classes}); got {tuple(t.shape)}"
        )
    if num_classes > 0:
        t = torch.clamp(t, 0, num_classes - 1)
    return t


class FeatureTokenBuilder(nn.Module):
    def __init__(
        self,
        groups: Iterable[Iterable[str]] | None,
        d_model: int,
        *,
        categorical: dict[str, int] | None = None,
        cat_agg: str = "sum",
    ) -> None:
        super().__init__()
        self.groups = normalize_feature_groups(groups)
        self.categorical: dict[str, int] = dict(DEFAULT_CATEGORICAL)
        if categorical:
            for k, v in categorical.items():
                self.categorical[str(k)] = int(v)
        self.cat_agg = str(cat_agg)
        if self.cat_agg not in {"sum", "mean"}:
            raise ValueError(f"cat_agg must be 'sum' or 'mean', got {cat_agg!r}")

        self.group_kinds: list[str] = []
        self.proj = nn.ModuleList()
        self.cat_embeds = nn.ModuleList()

        for group in self.groups:
            cat_names = [n for n in group if n in self.categorical]
            num_names = [n for n in group if n not in self.categorical]
            if cat_names and num_names:
                raise ValueError(
                    f"feature token group mixes categorical and numeric features: {group}"
                )
            if cat_names:
                self.group_kinds.append("cat")
                embeds = nn.ModuleDict(
                    {
                        name: nn.Embedding(int(self.categorical[name]), int(d_model))
                        for name in cat_names
                    }
                )
                self.cat_embeds.append(embeds)
                self.proj.append(nn.Identity())
            else:
                if not num_names:
                    raise ValueError("feature token group must not be empty")
                self.group_kinds.append("num")
                self.proj.append(nn.Linear(len(num_names), int(d_model)))
                self.cat_embeds.append(nn.ModuleDict())

    def feature_keys(self) -> list[str]:
        keys: list[str] = []
        for group in self.groups:
            keys.extend(group)
        return keys

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if not self.groups:
            return None
        tokens: list[torch.Tensor] = []
        for group, kind, proj, embeds in zip(
            self.groups, self.group_kinds, self.proj, self.cat_embeds, strict=True
        ):
            if kind == "num":
                parts = []
                for name in group:
                    if name not in features:
                        raise KeyError(f"Missing feature {name!r} in observation dict")
                    parts.append(_feature_tensor(features[name]).to(dtype=torch.float32))
                vec = torch.cat(parts, dim=1)
                tok = proj(vec).unsqueeze(1)
            else:
                cat_tokens: list[torch.Tensor] = []
                embeds_dict = cast(nn.ModuleDict, embeds)
                for name in group:
                    if name not in features:
                        raise KeyError(f"Missing feature {name!r} in observation dict")
                    num_classes = int(self.categorical[name])
                    ids = _as_cat_ids(features[name], num_classes)
                    emb = embeds_dict[name](ids)
                    cat_tokens.append(emb)
                if len(cat_tokens) == 1:
                    tok = cat_tokens[0].unsqueeze(1)
                else:
                    stacked = torch.stack(cat_tokens, dim=0)
                    fused = stacked.mean(dim=0) if self.cat_agg == "mean" else stacked.sum(dim=0)
                    tok = fused.unsqueeze(1)
            tokens.append(tok)
        return torch.cat(tokens, dim=1)
