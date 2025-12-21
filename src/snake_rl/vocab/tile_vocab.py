# src/snake_rl/vocab/tile_vocab.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from snake_rl.game.tile_types import TileType
from snake_rl.utils.paths import asset_path


@dataclass(frozen=True)
class TileVocab:
    """
    A compiled mapping from raw TileType.value IDs -> compact class IDs [0..K-1].

    - name: human-readable identifier (from YAML)
    - path: source YAML path
    - sha256: content hash for reproducibility
    - class_names: ordered list of class labels (defines IDs)
    - lut: numpy array of shape [raw_vocab_size], mapping raw_id -> class_id
    - num_classes: number of classes (K)
    """
    name: str
    path: Path
    sha256: str
    class_names: Tuple[str, ...]
    lut: np.ndarray
    num_classes: int

    def map_grid(self, raw_grid: np.ndarray) -> np.ndarray:
        """
        Map a raw tile-id grid (values == TileType.value) to class ids via LUT.

        Returns a view/copy depending on numpy advanced indexing rules.
        """
        if raw_grid.dtype != np.uint8 and raw_grid.dtype != np.int32 and raw_grid.dtype != np.int64:
            raw_grid = raw_grid.astype(np.int64, copy=False)
        return self.lut[raw_grid]


# --- internal cache (process-local) ---
_REGISTRY_CACHE: Optional[Dict[str, Path]] = None
_VOCAB_CACHE: Dict[str, TileVocab] = {}


def _assets_vocab_dir() -> Path:
    # Convention: put YAML vocab files into assets/vocabs/
    return asset_path("vocabs")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Tile vocab YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Tile vocab YAML top-level must be a dict, got {type(data).__name__}: {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_vocab_size() -> int:
    # Assumes TileType values are 0..K
    return int(max(int(t.value) for t in TileType) + 1)


def _all_yaml_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if root.is_dir():
        files.extend(sorted(root.glob("*.yaml")))
        files.extend(sorted(root.glob("*.yml")))
    return files


def _build_registry() -> Dict[str, Path]:
    """
    Scan assets/vocabs/*.ya?ml and return mapping: vocab_name -> file_path.

    The vocab_name is taken from the YAML field: `name: ...`

    Raises if:
      - a file is missing 'name'
      - the same name appears in multiple files
    """
    root = _assets_vocab_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"Vocab directory not found: {root}")

    by_name: Dict[str, Path] = {}
    collisions: Dict[str, List[Path]] = {}

    for p in _all_yaml_files(root):
        data = _read_yaml(p)
        name_v = data.get("name", None)
        if name_v is None:
            raise KeyError(f"Missing required key 'name' in tile vocab YAML: {p}")
        name = str(name_v).strip()
        if not name:
            raise ValueError(f"Tile vocab YAML has empty 'name': {p}")

        if name in by_name:
            collisions.setdefault(name, [by_name[name]]).append(p)
        else:
            by_name[name] = p

    if collisions:
        lines = ["Tile vocab name collision(s) detected:"]
        for name, paths in sorted(collisions.items(), key=lambda kv: kv[0]):
            ps = ", ".join(str(x) for x in paths)
            lines.append(f"  - name={name!r} in: {ps}")
        raise ValueError("\n".join(lines))

    if not by_name:
        raise FileNotFoundError(
            f"No tile vocab YAML files found in: {root} (expected *.yaml or *.yml)"
        )
    return by_name


def list_tile_vocabs() -> List[str]:
    """
    Return available vocab names from assets/vocabs.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return sorted(_REGISTRY_CACHE.keys())


def resolve_tile_vocab_path(name: str) -> Path:
    """
    Resolve a vocab by its YAML `name:` field, independent of filename.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()

    key = str(name).strip()
    if not key:
        raise ValueError("tile_vocab name must be a non-empty string")

    p = _REGISTRY_CACHE.get(key, None)
    if p is None:
        # Friendly suggestion list
        avail = list_tile_vocabs()
        preview = ", ".join(avail[:20])
        more = "" if len(avail) <= 20 else f" ... (+{len(avail) - 20} more)"
        raise KeyError(f"Unknown tile_vocab={key!r}. Available: {preview}{more}")
    return p


def _parse_classes(d: Any, *, ctx: str, path: Path) -> List[Tuple[str, List[TileType]]]:
    if not isinstance(d, dict):
        raise TypeError(f"Expected '{ctx}' to be a dict in {path}")

    out: List[Tuple[str, List[TileType]]] = []
    for class_name, members_v in d.items():
        cname = str(class_name).strip()
        if not cname:
            raise ValueError(f"Empty class name in {ctx} in {path}")

        if not isinstance(members_v, list) or not members_v:
            raise TypeError(f"Expected '{ctx}.{cname}' to be a non-empty list in {path}")

        members: List[TileType] = []
        for item in members_v:
            s = str(item).strip()
            if not s:
                raise ValueError(f"Empty TileType in '{ctx}.{cname}' in {path}")
            try:
                members.append(TileType[s])
            except KeyError as e:
                valid = ", ".join(t.name for t in TileType)
                raise ValueError(
                    f"Unknown TileType {s!r} in '{ctx}.{cname}' in {path}. "
                    f"Valid TileTypes: {valid}"
                ) from e

        out.append((cname, members))

    return out


def _compile_lut(*, classes: List[Tuple[str, List[TileType]]], path: Path) -> np.ndarray:
    raw_size = _raw_vocab_size()

    # Track coverage
    seen: Dict[TileType, str] = {}
    lut = np.zeros((raw_size,), dtype=np.uint8)

    for class_id, (cname, members) in enumerate(classes):
        for t in members:
            if t in seen:
                raise ValueError(
                    f"TileType {t.name} appears in multiple classes in {path}: "
                    f"{seen[t]!r} and {cname!r}"
                )
            seen[t] = cname
            lut[int(t.value)] = np.uint8(class_id)

    missing = [t for t in TileType if t not in seen]
    if missing:
        miss = ", ".join(t.name for t in missing)
        raise ValueError(f"Tile vocab in {path} is missing TileTypes: {miss}")

    # Also ensure no extras beyond enum (already guaranteed by parsing)
    return lut


def load_tile_vocab(name: str) -> TileVocab:
    """
    Load and compile a tile vocab by its internal YAML `name:`.

    This is cached per process. Use a unique name per vocab.
    """
    key = str(name).strip()
    if not key:
        raise ValueError("tile_vocab name must be a non-empty string")

    cached = _VOCAB_CACHE.get(key, None)
    if cached is not None:
        return cached

    path = resolve_tile_vocab_path(key)
    data = _read_yaml(path)

    # name must match request (guard against mismatched lookup)
    yaml_name = str(data.get("name", "")).strip()
    if yaml_name != key:
        raise ValueError(
            f"Tile vocab registry resolved name={key!r} to {path}, but YAML name is {yaml_name!r}. "
            f"Fix the YAML 'name' field or rename your request."
        )

    classes_v = data.get("classes", None)
    if classes_v is None:
        raise KeyError(f"Missing required key 'classes' in tile vocab YAML: {path}")

    classes = _parse_classes(classes_v, ctx="classes", path=path)
    if len(classes) < 2:
        raise ValueError(f"Tile vocab must define >= 2 classes, got {len(classes)} in {path}")

    lut = _compile_lut(classes=classes, path=path)
    class_names = tuple(cname for cname, _ in classes)

    sha = _sha256_file(path)

    vocab = TileVocab(
        name=key,
        path=path,
        sha256=sha,
        class_names=class_names,
        lut=lut,
        num_classes=len(class_names),
    )
    _VOCAB_CACHE[key] = vocab
    return vocab
