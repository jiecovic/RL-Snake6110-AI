# src/snake_rl/vocab/tile_vocab.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import snake_rl._core as core


@dataclass(frozen=True)
class TileVocab:
    """
    A compiled mapping from raw tile IDs -> compact class IDs [0..K-1].

    - name: human-readable identifier
    - source: provenance string (builtin today; Rust can own later)
    - sha256: stable hash of the expanded vocab definition
    - class_names: ordered list of class labels (defines IDs)
    - lut: numpy array of shape [raw_vocab_size], mapping raw_id -> class_id
    - num_classes: number of classes (K)
    """

    name: str
    source: str
    sha256: str
    class_names: tuple[str, ...]
    lut: np.ndarray
    num_classes: int

    def map_grid(self, raw_grid: np.ndarray) -> np.ndarray:
        """
        Map a raw tile-id grid (values == Rust tile ids) to class ids via LUT.

        Returns a view/copy depending on numpy advanced indexing rules.
        """
        if raw_grid.dtype not in (np.uint8, np.int32, np.int64):
            raw_grid = raw_grid.astype(np.int64, copy=False)
        return self.lut[raw_grid]


_VOCAB_CACHE: dict[str, TileVocab] = {}


def _all_tile_names() -> list[str]:
    return [str(x) for x in core.tileset_tile_names()]


def _normalize_classes(
    classes: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for cname, members in classes:
        cls = str(cname).strip()
        if not cls:
            raise ValueError("Tile vocab class name must be non-empty")
        if not members:
            raise ValueError(f"Tile vocab class {cls!r} has no members")
        out.append((cls, [str(x).strip() for x in members if str(x).strip()]))
    return out


def _compile_lut(classes: list[tuple[str, list[str]]]) -> np.ndarray:
    tile_names = _all_tile_names()
    name_to_id = {name: i for i, name in enumerate(tile_names)}
    raw_size = len(tile_names)

    seen: dict[int, str] = {}
    lut = np.zeros((raw_size,), dtype=np.uint8)

    for class_id, (cname, members) in enumerate(classes):
        for tname in members:
            if tname not in name_to_id:
                valid = ", ".join(tile_names)
                raise ValueError(
                    f"Unknown tile name {tname!r} in class {cname!r}. Valid tiles: {valid}"
                )
            tid = int(name_to_id[tname])
            if tid in seen:
                raise ValueError(
                    f"Tile id {tid} appears in multiple classes: {seen[tid]!r} and {cname!r}"
                )
            seen[tid] = cname
            lut[tid] = np.uint8(class_id)

    missing = [i for i in range(raw_size) if i not in seen]
    if missing:
        miss = ", ".join(tile_names[i] for i in missing)
        raise ValueError(f"Tile vocab is missing tiles: {miss}")

    return lut


def _sha256_vocab(name: str, classes: list[tuple[str, list[str]]]) -> str:
    payload = {
        "name": name,
        "classes": [(cname, members) for cname, members in classes],
    }
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _collect_vocab_defs() -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    ext = cast(Callable[[], Iterable[Mapping[str, Any]]], getattr(core, "tile_vocab_defs", None))
    if not callable(ext):
        raise RuntimeError("Rust core does not expose tile_vocab_defs()")

    defs: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    for item in ext():
        if not isinstance(item, dict):
            raise TypeError("tile_vocab_defs() items must be dicts")
        name = str(item.get("name", "")).strip()
        classes_v = item.get("classes", None)
        if not name:
            raise ValueError("tile_vocab_defs() item missing name")
        if classes_v is None:
            raise ValueError(f"tile_vocab_defs() missing classes for {name!r}")
        if name in defs:
            raise ValueError(f"Duplicate tile_vocab name from Rust: {name!r}")

        classes: list[tuple[str, tuple[str, ...]]] = []
        if isinstance(classes_v, dict):
            for cname, members in classes_v.items():
                classes.append((str(cname), tuple(str(x) for x in members)))
        else:
            for entry in classes_v:
                cname, members = entry
                classes.append((str(cname), tuple(str(x) for x in members)))

        defs[name] = tuple(classes)
    return defs


def list_tile_vocabs() -> list[str]:
    return sorted(_collect_vocab_defs().keys())


def load_tile_vocab(name: str) -> TileVocab:
    key = str(name).strip()
    if not key:
        raise ValueError("tile_vocab name must be a non-empty string")

    cached = _VOCAB_CACHE.get(key)
    if cached is not None:
        return cached

    defs = _collect_vocab_defs()
    classes_raw = defs.get(key)
    if classes_raw is None:
        avail = list_tile_vocabs()
        preview = ", ".join(avail[:20])
        more = "" if len(avail) <= 20 else f" ... (+{len(avail) - 20} more)"
        raise KeyError(f"Unknown tile_vocab={key!r}. Available: {preview}{more}")

    classes = _normalize_classes(classes_raw)
    lut = _compile_lut(classes)
    sha = _sha256_vocab(key, classes)
    class_names = tuple(cname for cname, _ in classes)

    vocab = TileVocab(
        name=key,
        source=f"rust:{key}",
        sha256=sha,
        class_names=class_names,
        lut=lut,
        num_classes=len(class_names),
    )
    _VOCAB_CACHE[key] = vocab
    return vocab


__all__ = ["TileVocab", "list_tile_vocabs", "load_tile_vocab"]
