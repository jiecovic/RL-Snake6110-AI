from core.snake6110.level import LevelFromTemplate, EmptyLevel


def print_level_info(level, name: str):
    print(f"\n=== {name} ===")
    print(level)

    try:
        snake = level.get_snake_segments()
        print(f"Snake segments: {snake}")
        print(f"Snake head direction: {level.get_snake_direction()}")
    except ValueError as e:
        print(f"No valid snake found: {e}")

    walls = level.get_wall_tiles()
    print(f"Wall tiles:")
    for tile in walls:
        print(tile)

    food = level.get_food_positions()
    print(f"Food positions: {food}")


def main():
    level1 = LevelFromTemplate("assets/levels/test_level.json")
    level2 = EmptyLevel(width=20, height=11)

    print_level_info(level1, "Loaded Level")
    print_level_info(level2, "Generated Level")

    level2.to_json("assets/levels/generated_output.json")


if __name__ == "__main__":
    main()
