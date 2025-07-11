import pygame
import random
from typing import Optional, List
from enum import Enum, auto
from core.snake6110.level import BaseLevel
from core.snake6110.geometry import Point, Direction, DIRECTION_TO_POINT, RelativeDirection
from core.snake6110.tile_types import is_wall, is_food, is_tail, TileType


class MoveResult(Enum):
    OK = auto()  # Normal move, nothing special
    FOOD_EATEN = auto()  # Snake ate food
    HIT_BOUNDARY = auto()  # Moved outside grid
    HIT_WALL = auto()  # Hit a wall tile
    HIT_SELF = auto()  # Ran into its own body
    GAME_NOT_RUNNING = auto()  # Called move() after game over


class SnakeGame:
    def __init__(self, level: BaseLevel, food_count: Optional[int] = None, fps: int = 5, tile_size: int = 32):
        self.level = level
        self.width = level.width
        self.height = level.height
        self.tile_size = tile_size
        self.fps = fps

        self.snake = level.get_snake_segments()
        self.direction = level.get_snake_direction()
        self.running = True

        existing_food = level.get_food_positions()

        print(f"EXISTING FOOD: {existing_food} ({len(existing_food)})")

        # Always use all foods from template at init
        self.food = list(existing_food)

        # Determine upper bound for food
        if food_count is not None:
            self.target_food_count = food_count
        elif existing_food:
            self.target_food_count = len(existing_food)
        else:
            raise ValueError("No food defined in template and no food_count specified.")

        print(f"TARGET FOOD COUNT: {self.target_food_count}")
        print(f"FINAL INITIAL FOOD: {self.food}")

        # === Initialize food list ===
        # self.food = list(existing_food[:self.target_food_count])
        while len(self.food) < self.target_food_count:
            self._spawn_food()

        # Init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * tile_size, self.height * tile_size))
        self.clock = pygame.time.Clock()

    def _spawn_food(self):
        max_attempts = 100
        for _ in range(max_attempts):
            p = Point(random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if p not in self.snake and p not in self.food and not is_wall(self.level.get_tile_type(p)):
                self.food.append(p)
                return
        print("Warning: Could not find suitable food spawn location.")

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> MoveResult:
        if not self.running:
            return MoveResult.GAME_NOT_RUNNING

        # Update direction
        if rel_dir == RelativeDirection.LEFT:
            self.direction = self.direction.turn_left()
        elif rel_dir == RelativeDirection.RIGHT:
            self.direction = self.direction.turn_right()

        vec = DIRECTION_TO_POINT[self.direction]
        new_head = Point(self.snake[0].x + vec.x, self.snake[0].y + vec.y)

        # Collision checks
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.running = False
            print("Game over: hit boundary.")
            return MoveResult.HIT_BOUNDARY

        if is_wall(self.level.get_tile_type(new_head)):
            self.running = False
            print("Game over: hit wall.")
            return MoveResult.HIT_WALL

        if new_head in self.snake:
            self.running = False
            print("Game over: hit itself.")
            return MoveResult.HIT_SELF

        # Move
        self.snake.insert(0, new_head)
        if new_head in self.food:
            self.food.remove(new_head)
            while len(self.food) < self.target_food_count:
                self._spawn_food()
            return MoveResult.FOOD_EATEN
        else:
            self.snake.pop()
            return MoveResult.OK

    def render(self) -> None:
        self.screen.fill((0, 0, 0))

        # Draw walls
        for y in range(self.height):
            for x in range(self.width):
                tile = self.level.get_tile_type(Point(x, y))
                if is_wall(tile):
                    pygame.draw.rect(self.screen, (128, 128, 128),
                                     (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))

        # Draw food
        for food in self.food:
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (food.x * self.tile_size, food.y * self.tile_size, self.tile_size, self.tile_size))

        # Draw snake
        for i, part in enumerate(self.snake):
            color = (0, 255, 0) if i > 0 else (0, 200, 255)  # head vs body
            pygame.draw.rect(self.screen, color,
                             (part.x * self.tile_size, part.y * self.tile_size, self.tile_size, self.tile_size))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
