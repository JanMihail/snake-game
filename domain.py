import copy
import random
import threading
import time
from enum import Enum

from termcolor import colored


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def inc_x(self):
        self._x += 1

    def dec_x(self):
        self._x -= 1

    def inc_y(self):
        self._y += 1

    def dec_y(self):
        self._y -= 1

    def __eq__(self, o) -> bool:
        return self._x == o.get_x() and self._y == o.get_y()

    def __hash__(self) -> int:
        return hash((self._x, self._y))

    def __str__(self) -> str:
        return "x = " + str(self._x) + ", y = " + str(self._y)


class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class GameState(Enum):
    READY = 1
    PLAYING = 2
    GAME_OVER = 3


class Snake:
    def __init__(self, head_x: int, head_y: int, length: int, direction: Direction):
        self._foodCounter = 0
        self._direction = direction
        self._last_step_direction = direction

        # Create head
        head = Point(head_x, head_y)
        self._points = [head]

        # Create body
        i = 1
        prev = head
        while i < length:
            p_next = copy.deepcopy(prev)

            if direction == Direction.LEFT:
                p_next.inc_x()
            elif direction == Direction.RIGHT:
                p_next.dec_x()
            elif direction == Direction.UP:
                p_next.dec_y()
            elif direction == Direction.DOWN:
                p_next.inc_y()
            else:
                raise Exception("Unknown direction")

            self._points.append(p_next)
            prev = p_next
            i += 1

    def set_direction(self, new_direction: Direction):
        # Deny change direction UP <-> DOWN и RIGHT <-> LEFT
        if self._last_step_direction.value + new_direction.value not in [3, 7]:
            self._direction = new_direction
        else:
            print(colored(
                f"Incorrect direction. Last step: {self._last_step_direction}. New: {new_direction}",
                "yellow"
            ))

    def feed(self):
        self._foodCounter += 1

    def make_step(self):
        new_head = copy.deepcopy(self._points[0])

        if self._direction == Direction.LEFT:
            new_head.dec_x()
        elif self._direction == Direction.RIGHT:
            new_head.inc_x()
        elif self._direction == Direction.UP:
            new_head.inc_y()
        elif self._direction == Direction.DOWN:
            new_head.dec_y()
        else:
            raise Exception("Unknown direction")

        self._last_step_direction = self._direction

        self._points.insert(0, new_head)

        if self._foodCounter > 0:
            self._foodCounter -= 1
        else:
            self._points.pop()

    def exist_circle_collision(self) -> bool:
        return len(self._points) != len(set(self._points))

    def collision_with_point(self, p: Point) -> bool:
        for body in self._points:
            if body == p:
                return True
        return False

    def get_head(self) -> Point:
        return copy.deepcopy(self._points[0])

    def get_points(self) -> list:
        return self._points

    def __str__(self) -> str:
        res = "Snake\n"

        res += "  points: \n"
        for p in self._points:
            res += "    [" + str(p) + "]" + "\n"

        res += "  direction: " + self._direction.name + "\n"
        return res


class BlocksBuilder:
    @staticmethod
    def create_rect(width: int, height: int) -> list:
        x = 0
        blocks = []
        while x < width:
            blocks.append(Point(x, 0))
            blocks.append(Point(x, height - 1))
            x += 1

        y = 0
        while y < height:
            blocks.append(Point(0, y))
            blocks.append(Point(width - 1, y))
            y += 1

        return blocks


class GameEngine:
    def __init__(self, width: int, height: int, delay: float):
        self._width = width
        self._height = height
        self._delay = delay
        self._score = 0
        self._state = GameState.READY
        self._blocks = BlocksBuilder.create_rect(width, height)
        self._snake = Snake(10, 10, 3, Direction.UP)
        self._food = self.__create_food()

    def restart(self):
        self._score = 0
        self._state = GameState.READY
        self._blocks = BlocksBuilder.create_rect(self._width, self._height)
        self._snake = Snake(10, 10, 3, Direction.UP)
        self._food = self.__create_food()
        self.start()

    def start(self):
        self._state = GameState.PLAYING

        t = threading.Thread(target=self.__gameloop, daemon=True)
        t.start()
        print("Game started...")

    def get_score(self) -> int:
        return self._score

    def get_snake_coords(self) -> list[Point]:
        return copy.deepcopy(self._snake.get_points())

    def get_blocks(self) -> list[Point]:
        return self._blocks

    def get_food(self) -> Point:
        return self._food

    def get_state(self) -> GameState:
        return self._state

    def change_snake_direction(self, new_direction: Direction):
        self._snake.set_direction(new_direction)

    def __gameloop(self):
        while self._state != GameState.GAME_OVER:
            self._snake.make_step()

            if self.__exist_colission_with_block() or self._snake.exist_circle_collision():
                self._state = GameState.GAME_OVER
                continue

            if self.__exist_colission_with_food():
                self._snake.feed()
                self._score += 1
                self._food = self.__create_food()

            time.sleep(self._delay)

        print(f"Game over! Score: {self._score}")

    def __exist_colission_with_block(self) -> bool:
        head = self._snake.get_head()
        for block in self._blocks:
            if block == head:
                return True
        return False

    def __exist_colission_with_food(self) -> bool:
        return self._food == self._snake.get_head()

    def __create_food(self) -> Point:
        while True:
            x = random.randint(1, self._width - 2)
            y = random.randint(1, self._height - 2)
            food = Point(x, y)

            if not self._snake.collision_with_point(food):
                return food
