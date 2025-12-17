# src/snake_rl/tools/print_level.py
from __future__ import annotations

from snake_rl.game.level import EmptyLevel, TemplateLevel
from snake_rl.utils.paths import repo_root


def print_level_info(level, *, name: str) -> None:
    print(f"\n=== {name} ===")
    print(level)

    try:
        snake = level.get_snake_segments()
        print(f"Snake segments: {snake}")
        print(f"Snake head direction: {level.get_snake_direction()}")
    except ValueError as e:
        print(f"No valid snake found: {e}")

    walls = level.get_wall_tiles()
    print("Wall tiles:")
    for tile in walls:
        print(tile)

    food = level.get_food_positions()
    print(f"Food positions: {food}")


def main() -> None:
    root = repo_root()
    levels_dir = root / "assets" / "levels"

    level1_path = levels_dir / "test_level.yaml"
    out_path = levels_dir / "generated_output.yaml"

    level1 = TemplateLevel(level1_path)
    level2 = EmptyLevel(width=20, height=11)

    print_level_info(level1, name="Loaded Level")
    print_level_info(level2, name="Generated Level")

    level2.to_yaml(out_path)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
