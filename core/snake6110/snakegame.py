import pygame
import random
import numpy as np
from typing import Optional, List, Union, Dict
from enum import Enum, auto
from core.snake6110.level import BaseLevel
from core.snake6110.geometry import Point, Direction, DIRECTION_TO_POINT, RelativeDirection
from core.snake6110.tile_types import is_wall, is_food, is_tail, TileType
from core.snake6110.tileset import Tileset

np.set_printoptions(threshold=np.inf, linewidth=100000)  # No truncation of large arrays


class MoveResult(Enum):
    OK = auto()               # Normal move, nothing special
    FOOD_EATEN = auto()       # Snake ate food
    HIT_BOUNDARY = auto()     # Moved outside grid
    HIT_WALL = auto()         # Hit a wall tile
    HIT_SELF = auto()         # Ran into its own body
    GAME_NOT_RUNNING = auto() # Called move() after game over
    TIMEOUT = auto()          # Max steps reached (episode truncated)



class SnakeGame:
    def __init__(self, level: BaseLevel, food_count: Optional[int] = None, fps: int = 1, pixel_size: int = 10, tileset: Optional[Tileset] = None, seed: Optional[int] = None):
        # === Game settings ===
        self.level = level
        self.width = level.width
        self.height = level.height
        self.tileset = tileset or Tileset()
        self.pixel_size = pixel_size
        self.fps = fps

        # === RNG ===
        self.seed_value = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # === Tile rendering size ===
        self.tile_dim = self.tileset.tile_size * self.pixel_size
        print(f"[INIT] TILE_SIZE={self.tileset.tile_size}, PIXEL_SIZE={self.pixel_size}, TILE_DIM={self.tile_dim}")

        # === Pixel buffer (1-channel grayscale) ===
        self.pixel_buffer = np.zeros((self.height * self.tileset.tile_size, self.width * self.tileset.tile_size), dtype=np.uint8)

        # === PyGame ===
        self.render_enabled = False
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None

        # === Wall definitions ===
        self.wall_positions = {pos for pos, _ in level.get_wall_tiles()}
        self.wall_tiles = level.get_wall_tiles()

        # === Wall layer (1-channel binary layer) ===
        self.wall_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

        # === Food & Snake layers ===
        self.snake_layers: Dict[TileType, np.ndarray] = {}
        self.food_layer = np.zeros((self.height, self.width), dtype=np.uint8)

        # === Food target count ===
        existing_food = level.get_food_positions()
        if food_count is not None:
            self.target_food_count = food_count
        elif existing_food:
            self.target_food_count = len(existing_food)
        else:
            raise ValueError("No food defined in level and no food_count specified.")

        # print(f"[INIT] TARGET_FOOD_COUNT={self.target_food_count}")

        # === Initialize game state ===
        self.reset()

    def set_seed(self, seed: Optional[int]) -> None:
        """Set or reset the RNG for reproducibility."""
        self.seed_value = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset snake, food, score, layers, and running state"""
        # === Reset snake ===
        self.snake = self.level.get_snake_segments()
        self.direction = self.level.get_snake_direction()

        # === Reset food ===
        self.food = list(self.level.get_food_positions())
        while len(self.food) < self.target_food_count:
            self._spawn_food()

        # === Reset state ===
        self.score = 0
        self.running = True

        # === Reset pixe buffer ===
        self.pixel_buffer.fill(0)

        # === Update layer maps ===
        self._update_wall_layer()
        self._update_snake_layers()
        self._update_food_layer()

    def get_head_position(self) -> Point:
        """
        Returns the current position of the snake's head.
        """
        return self.snake[0]

    def get_food_positions(self) -> List[Point]:
        """
        Returns the current list of food positions.
        """
        return self.food.copy()


    def _spawn_food(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            p = Point(
                self.rng.randint(0, self.width - 1),
                self.rng.randint(0, self.height - 1)
            )
            if p not in self.snake and p not in self.food and not is_wall(self.level.get_tile_type(p)):
                self.food.append(p)
                return

    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> MoveResult:
        result = self._move(rel_dir)
        self._update_snake_layers()
        self._update_pixel_buffer()
        return result

    def _move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> MoveResult:
        if not self.running:
            return MoveResult.GAME_NOT_RUNNING

        # Compute tentative new direction (but don't apply it yet)
        new_direction = self.direction
        if rel_dir == RelativeDirection.LEFT:
            new_direction = self.direction.turn_left()
        elif rel_dir == RelativeDirection.RIGHT:
            new_direction = self.direction.turn_right()

        vec = DIRECTION_TO_POINT[new_direction]
        new_head = Point(self.snake[0].x + vec.x, self.snake[0].y + vec.y)

        # Collision checks
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.running = False
            return MoveResult.HIT_BOUNDARY

        if new_head in self.wall_positions:
            self.running = False
            return MoveResult.HIT_WALL

        if new_head in self.snake:
            self.running = False
            return MoveResult.HIT_SELF

        # All clear: apply direction
        self.direction = new_direction

        # Move
        self.snake.insert(0, new_head)
        if new_head in self.food:
            self.food.remove(new_head)
            self.score += 1  # <-- Add this line
            while len(self.food) < self.target_food_count:
                self._spawn_food()
            return MoveResult.FOOD_EATEN
        else:
            self.snake.pop()
            return MoveResult.OK

    def _update_wall_layer(self) -> None:
        """Update the binary wall layer based on static wall positions."""
        self.wall_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

    def _update_food_layer(self) -> None:
        self.food_layer.fill(0)
        for food in self.food:
            self.food_layer[food.y, food.x] = 1

    def _update_snake_layers(self) -> None:
        self.snake_layers = {
            tile_type: np.zeros((self.height, self.width), dtype=np.float32)
            for tile_type in TileType
            if tile_type.name.startswith("SNAKE_")
        }

        if len(self.snake) < 2:
            return

        # Head
        head = self.snake[0]
        head_type = {
            Direction.UP: TileType.SNAKE_HEAD_UP,
            Direction.DOWN: TileType.SNAKE_HEAD_DOWN,
            Direction.LEFT: TileType.SNAKE_HEAD_LEFT,
            Direction.RIGHT: TileType.SNAKE_HEAD_RIGHT
        }[self.direction]
        self.snake_layers[head_type][head.y, head.x] = 1.0

        # Body
        for i in range(1, len(self.snake) - 1):
            prev = self.snake[i - 1]
            curr = self.snake[i]
            next_ = self.snake[i + 1]

            if prev.x == next_.x:
                tile_type = TileType.SNAKE_BODY_VERTICAL
            elif prev.y == next_.y:
                tile_type = TileType.SNAKE_BODY_HORIZONTAL
            elif ((prev.x < curr.x and next_.y > curr.y) or
                  (next_.x < curr.x and prev.y > curr.y)):
                tile_type = TileType.SNAKE_BODY_TR
            elif ((prev.x > curr.x and next_.y > curr.y) or
                  (next_.x > curr.x and prev.y > curr.y)):
                tile_type = TileType.SNAKE_BODY_TL
            elif ((prev.x < curr.x and next_.y < curr.y) or
                  (next_.x < curr.x and prev.y < curr.y)):
                tile_type = TileType.SNAKE_BODY_BR
            elif ((prev.x > curr.x and next_.y < curr.y) or
                  (next_.x > curr.x and prev.y < curr.y)):
                tile_type = TileType.SNAKE_BODY_BL
            else:
                raise RuntimeError(f"Could not determine body tile type at index {i}: prev={prev}, curr={curr}, next={next_}")

            self.snake_layers[tile_type][curr.y, curr.x] = 1.0

        # Tail
        tail = self.snake[-1]
        prev = self.snake[-2]
        if prev.x < tail.x:
            tail_type = TileType.SNAKE_TAIL_RIGHT
        elif prev.x > tail.x:
            tail_type = TileType.SNAKE_TAIL_LEFT
        elif prev.y < tail.y:
            tail_type = TileType.SNAKE_TAIL_DOWN
        elif prev.y > tail.y:
            tail_type = TileType.SNAKE_TAIL_UP
        else:
            raise RuntimeError(f"Could not determine tail direction: prev={prev}, tail={tail}")

        self.snake_layers[tail_type][tail.y, tail.x] = 1.0

    def get_all_layers(self, channel_last: bool = False, merge_snake_layers: bool = False) -> np.ndarray:
        """
        Returns all game layers (snake, wall, food) as a stacked NumPy array.

        Args:
            channel_last (bool): If True, returns shape (H, W, C). Else, (C, H, W).
            merge_snake_layers (bool): If True, merges all snake layers into a single channel.

        Returns:
            np.ndarray: The combined layer tensor.
        """
        if merge_snake_layers:
            snake_combined = np.zeros((self.height, self.width), dtype=np.float32)
            for layer in self.snake_layers.values():
                snake_combined += layer
            layers = [snake_combined]
        else:
            layers = list(self.snake_layers.values())

        layers.append(self.wall_layer)
        layers.append(self.food_layer)

        stacked = np.stack(layers, axis=0)  # shape: (C, H, W)

        if channel_last:
            stacked = np.transpose(stacked, (1, 2, 0))  # shape: (H, W, C)

        return stacked

    def _update_pixel_buffer(self) -> None:
        """
        Updates the pixel buffer (`self.pixel_buffer`) as a 2D NumPy array of shape
        (height * tile_size, width * tile_size), where each tile is drawn without additional upscaling.
        """
        tile_dim = self.tileset.tile_size
        pixel_height = self.height * tile_dim
        pixel_width = self.width * tile_dim

        self.pixel_buffer = np.zeros((pixel_height, pixel_width), dtype=np.uint8)

        # === Draw walls ===
        for point, tile_type in self.wall_tiles:
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

        # === Draw food ===
        if TileType.FOOD in self.tileset:
            tile = np.array(self.tileset[TileType.FOOD], dtype=np.uint8)
            for point in self.food:
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

        # === Draw snake ===
        for tile_type, layer in self.snake_layers.items():
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                yxs = np.argwhere(layer > 0)
                for y, x in yxs:
                    py, px = y * tile_dim, x * tile_dim
                    self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

    def _init_renderer(self):
        if self.pygame_initialized:
            return
        pygame.init()
        width = self.width * self.tile_dim
        height = self.height * self.tile_dim
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.pygame_initialized = True
        self.render_enabled = True

    def render(self) -> None:
        """
        Public method to render the game. Initializes pygame lazily.
        """
        if not self.pygame_initialized:
            self._init_renderer()

        if self.pixel_buffer is None:
            return  # nothing to draw yet

        self._draw()

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.clock.tick(self.fps)

    def _draw(self) -> None:
        """
        Internal method that performs the actual drawing logic.
        Assumes pygame is initialized and pixel_buffer is available.
        """
        if self.pixel_buffer is None:
            return

        # 1. Upscale pixel_buffer (2D grayscale) using np.kron
        upscaled = np.kron(
            self.pixel_buffer,
            np.ones((self.pixel_size, self.pixel_size), dtype=self.pixel_buffer.dtype)
        )

        # 2. Convert to RGB by stacking the grayscale values 3 times
        rgb_buffer = np.stack([upscaled * 255] * 3, axis=-1).astype(np.uint8)

        # 3. Convert to surface and draw
        surface = pygame.surfarray.make_surface(rgb_buffer.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        # 4. Draw score text
        score_surf = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))  # position fixed since tile_dim no longer used

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
