# src/snake_rl/game/rendering/pygame/hud.py

from __future__ import annotations

from typing import Any

from snake_rl.envs.specs import ObservationSpec
from snake_rl.game.snake_engine import SnakeEngine


def hud_feature_flags(*, hud_mode: str, hud_features: dict[str, Any]) -> dict[str, Any]:
    mode = str(hud_mode).strip().lower()
    if mode == "selected":
        return {k: v for k, v in hud_features.items() if v}
    return {
        "direction": True,
        "snake_progress": True,
        "time_since_food": True,
        "closest_food": True,
        "collision_ahead": True,
        "collision_left": True,
        "collision_right": True,
    }


def feature_items(
    *, game: SnakeEngine, flags: dict[str, Any], include_snake_progress: bool = True
) -> list[tuple[str, str]]:
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
        metric = closest_food_metric(flags.get("closest_food"))
        dx, dy, dist = game.get_closest_food_norm(metric)
        items.append(("Food", f"dx {dx:+.2f} dy {dy:+.2f} d {dist:.2f}"))

    if flags.get("collision_ahead") or flags.get("collision_left") or flags.get("collision_right"):
        ahead, left, right = game.collision_flags()
        items.append(("Coll", f"A {int(ahead)} L {int(left)} R {int(right)}"))

    return items


def closest_food_metric(value: Any) -> str:
    if isinstance(value, dict):
        metric = value.get("metric")
        if metric is not None:
            return str(metric)
    if isinstance(value, str):
        return str(value)
    return "manhattan"


def info_pairs(
    *, hud_info: dict[str, Any], agent_view_spec: ObservationSpec | None
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    mode = str(hud_info.get("mode", "")).strip()
    if mode:
        items.append(("Mode", mode))
    run = str(hud_info.get("run", "")).strip()
    if run:
        items.append(("Run", run))
    which = str(hud_info.get("which", "")).strip()
    if which:
        items.append(("Which", which))
    seed = hud_info.get("seed")
    if seed is not None and str(seed).strip():
        items.append(("Seed", str(seed)))
    obs = hud_info.get("obs")
    if obs is None and agent_view_spec is not None:
        kind = agent_view_spec.kind_norm()
        view = agent_view_spec.view_norm()
        obs = f"{kind}/{view}"
    if obs:
        items.append(("Obs", str(obs)))
    action = str(hud_info.get("action", "")).strip()
    if action:
        items.append(("Action", action))
    return items


def format_pairs(
    pairs: list[tuple[str, str]],
    *,
    empty_text: str | None = None,
    key_pad: int = 0,
) -> list[str]:
    if not pairs:
        return [str(empty_text)] if empty_text is not None else []
    key_w = max(len(k) for k, _v in pairs) + max(0, int(key_pad))
    return [f"{k:<{key_w}} {v}" for k, v in pairs]


def format_section(
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
