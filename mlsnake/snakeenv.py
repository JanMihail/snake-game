import copy

import numpy as np
import tkinter as tk
from gym import Env, spaces
from gym.utils import seeding

from domain import Point, Snake, Direction, BlocksBuilder


class Config:
    DELAY = 10
    WIDTH = 600
    HEIGHT = 600
    STEP_X = 30
    STEP_Y = 30
    DELTA_XY = 2


class GameBoard(tk.Canvas):
    def __init__(self, blocks: list[Point]):
        super(GameBoard, self).__init__(background="#F7E697", width=Config.WIDTH, height=Config.HEIGHT)

        self._img_block = tk.PhotoImage(file="../resources/block.png")
        self._img_body = tk.PhotoImage(file="../resources/body.png")
        self._img_head = tk.PhotoImage(file="../resources/head.png")
        self._img_food = tk.PhotoImage(file="../resources/food.png")

        self.__init_board(blocks)

    def __init_board(self, blocks: list[Point]):
        for block in blocks:
            self.create_image(
                self.__x_to_screen(block.get_x()),
                self.__y_to_screen(block.get_y()),
                image=self._img_block,
                anchor='nw'
            )
        self.update()

    def redraw(self, food: Point, snake_blocks: list[Point], score: int):
        self.__redraw_food(food)
        self.__redraw_snake(snake_blocks)
        self.__redraw_score(score)
        self.update()

    def __redraw_food(self, food: Point):
        self.delete('food')
        self.create_image(
            self.__x_to_screen(food.get_x()),
            self.__y_to_screen(food.get_y()),
            image=self._img_food,
            anchor='nw',
            tag='food'
        )

    def __redraw_snake(self, snake_blocks: list[Point]):
        self.delete("snake")

        head = snake_blocks.pop(0)

        for body in snake_blocks:
            self.create_image(
                self.__x_to_screen(body.get_x()),
                self.__y_to_screen(body.get_y()),
                image=self._img_body,
                anchor='nw',
                tag='snake'
            )

        self.create_image(
            self.__x_to_screen(head.get_x()),
            self.__y_to_screen(head.get_y()),
            image=self._img_head,
            anchor='nw',
            tag='snake'
        )

    def __redraw_score(self, score: int):
        self.delete('score')
        msg = "Score: " + str(score)
        self.create_text(5, 5, text=msg, font=('Consolas', 14), anchor='nw', tags='score')

    def __x_to_screen(self, x):
        return Config.STEP_X * x + Config.DELTA_XY

    def __y_to_screen(self, y):
        return Config.HEIGHT - Config.STEP_Y - Config.STEP_Y * y + Config.DELTA_XY


class SnakeViewer(tk.Tk):
    def __init__(self, blocks: list[Point]):
        super().__init__()

        self.gb = GameBoard(blocks)
        self.gb.pack()

        self.wm_title("Snake Super Game")
        self.eval('tk::PlaceWindow . center')
        self.resizable(False, False)

        self.gb.update()


class SnakeEnv(Env):

    def __init__(self) -> None:
        super().__init__()

        self._viewer = None

        self._action_map = {0: Direction.UP,
                            1: Direction.RIGHT,
                            2: Direction.DOWN,
                            3: Direction.LEFT}
        self.state = None
        self.np_random = None
        self.seed()

        self.observation_shape = (12,)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape, dtype=np.float32),
                                            high=np.ones(self.observation_shape, dtype=np.float32),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._width = 20
        self._height = 20
        self._blocks = BlocksBuilder.create_rect(self._width, self._height)
        self.reset()

    # noinspection PyAttributeOutsideInit
    def step(self, action: int):
        done = False
        new_direction = self._action_map.get(action)
        prev_head = self._snake.get_head()

        self._snake.set_direction(new_direction)
        self._snake.make_step()

        if self.__exist_colission_with_block() or self._snake.exist_circle_collision():
            reward = -200.
            done = True

        elif self.__exist_colission_with_food():
            self._snake.feed()
            self._food = self.__create_food()
            reward = 40.
            self._score += 1

        elif self.__comes_to_food(prev_head, self._snake.get_head(), self._food):
            reward = 1.

        else:
            reward = -1.

        return self._get_ob(), reward, done, {}

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self._snake = Snake(10, 10, 3, Direction.UP)
        self._food = self.__create_food()
        self._score = 0

        return self._get_ob()

    def render(self, mode="human"):
        if self._viewer is None:
            self._viewer = SnakeViewer(self._blocks)

        self._viewer.gb.redraw(
            self._food,
            copy.deepcopy(self._snake.get_points()),
            self._score
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        super().close()

    def _get_ob(self):
        head = self._snake.get_head()

        all_blocks = self._snake.get_points()[1:]
        all_blocks += self._blocks

        block_up = 0.
        block_right = 0.
        block_down = 0.
        block_left = 0.

        for block in all_blocks:
            if block_up < 0.01 and block.get_y() == head.get_y() + 1:
                block_up = 1.
            if block_right < 0.01 and block.get_x() == head.get_x() + 1:
                block_right = 1.
            if block_down < 0.01 and block.get_y() == head.get_y() - 1:
                block_down = 1.
            if block_left < 0.01 and block.get_x() == head.get_x() - 1:
                block_left = 1.

        food_up = 1. if self._food.get_y() > head.get_y() else 0.
        food_right = 1. if self._food.get_x() > head.get_x() else 0.
        food_down = 1. if self._food.get_y() < head.get_y() else 0.
        food_left = 1. if self._food.get_x() < head.get_x() else 0.

        dir_up = 1. if self._snake.get_direction() == Direction.UP else 0.
        dir_right = 1. if self._snake.get_direction() == Direction.RIGHT else 0.
        dir_down = 1. if self._snake.get_direction() == Direction.DOWN else 0.
        dir_left = 1. if self._snake.get_direction() == Direction.LEFT else 0.

        return np.array(
            [
                food_up, food_right, food_down, food_left,
                block_up, block_right, block_down, block_left,
                dir_up, dir_right, dir_down, dir_left
            ],
            dtype=np.float32
        )

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
            x = self.np_random.randint(1, self._width - 1)
            y = self.np_random.randint(1, self._height - 1)
            food = Point(x, y)

            if not self._snake.collision_with_point(food):
                return food

    def __comes_to_food(self, prev_head: Point, current_head: Point, food: Point):
        prev_distance = abs(food.get_x() - prev_head.get_x()) + (food.get_y() - prev_head.get_y())
        curr_distance = abs(food.get_x() - current_head.get_x()) + (food.get_y() - current_head.get_y())
        return curr_distance < prev_distance
