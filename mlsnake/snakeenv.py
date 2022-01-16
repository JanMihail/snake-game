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

        self.observation_shape = (5,)
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
        reward = 1.0
        new_direction = self._action_map.get(action)

        self._snake.set_direction(new_direction)
        self._snake.make_step()

        if self.__exist_colission_with_block() or self._snake.exist_circle_collision():
            done = True

        elif self.__exist_colission_with_food():
            self._snake.feed()
            self._food = self.__create_food()
            reward = 2.0

        self._score += reward

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

        head_up = None
        head_right = None
        head_down = None
        head_left = None

        all_blocks = self._snake.get_points()[1:]
        all_blocks += self._blocks

        for block in all_blocks:
            if head.get_x() == block.get_x():
                # up
                if head.get_y() - block.get_y() < 0:
                    if head_up is None or head_up.get_y() - block.get_y() > 0:
                        head_up = block
                # down
                else:
                    if head_down is None or head_down.get_y() - block.get_y() < 0:
                        head_down = block
            elif head.get_y() == block.get_y():
                # left
                if head.get_x() - block.get_x() > 0:
                    if head_left is None or head_left.get_x() - block.get_x() < 0:
                        head_left = block
                # right
                else:
                    if head_right is None or head_right.get_x() - block.get_x() > 0:
                        head_right = block

        if head_left is None:
            head_left = Point(head.get_x(), head.get_y())
        if head_right is None:
            head_right = Point(head.get_x(), head.get_y())
        if head_up is None:
            head_up = Point(head.get_x(), head.get_y())
        if head_down is None:
            head_down = Point(head.get_x(), head.get_y())

        up_dist = (head_up.get_y() - head.get_y()) / (self._height - 2)
        down_dist = (head.get_y() - head_down.get_y()) / (self._height - 2)
        right_dist = (head_right.get_x() - head.get_x()) / (self._width - 2)
        left_dist = (head.get_x() - head_left.get_x()) / (self._width - 2)

        return np.array(
            [up_dist, right_dist, down_dist, left_dist, 0.], dtype=np.float32
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
