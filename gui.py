import tkinter as tk

from domain import GameEngine, Direction, GameState


class Config:
    DELAY = 10
    WIDTH = 600
    HEIGHT = 600
    STEP_X = 30
    STEP_Y = 30
    DELTA_XY = 2


class GameBoard(tk.Canvas):
    def __init__(self, master, game_engine: GameEngine):
        super(GameBoard, self).__init__(master, background="#F7E697", width=Config.WIDTH, height=Config.HEIGHT)
        self._game_engine = game_engine

        self._img_block = tk.PhotoImage(file="resources/block.png")
        self._img_body = tk.PhotoImage(file="resources/body.png")
        self._img_head = tk.PhotoImage(file="resources/head.png")
        self._img_food = tk.PhotoImage(file="resources/food.png")

        self.__init_board()

        self.bind_all("<Key>", self.on_key_pressed)
        self.after(Config.DELAY, self.on_timer)

    def on_key_pressed(self, e):
        key = e.keysym

        if key == "Left":
            self._game_engine.change_snake_direction(Direction.LEFT)
        elif key == "Right":
            self._game_engine.change_snake_direction(Direction.RIGHT)
        elif key == "Up":
            self._game_engine.change_snake_direction(Direction.UP)
        elif key == "Down":
            self._game_engine.change_snake_direction(Direction.DOWN)
        elif key.lower() == "r" and self._game_engine.get_state() == GameState.GAME_OVER:
            self._game_engine.restart()

    def on_timer(self):
        self.__update()
        self.after(Config.DELAY, self.on_timer)

    def __init_board(self):
        blocks = self._game_engine.get_blocks()

        for block in blocks:
            self.create_image(
                self.__x_to_screen(block.get_x()),
                self.__y_to_screen(block.get_y()),
                image=self._img_block,
                anchor='nw'
            )

    def __update(self):
        self.__update_food()
        self.__update_snake()
        self.__update_score()

    def __update_food(self):
        self.delete('food')
        food = self._game_engine.get_food()
        self.create_image(
            self.__x_to_screen(food.get_x()),
            self.__y_to_screen(food.get_y()),
            image=self._img_food,
            anchor='nw',
            tag='food'
        )

    def __update_snake(self):
        self.delete("snake")

        snake_blocks = self._game_engine.get_snake_coords()

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

    def __update_score(self):
        self.delete('score')
        msg = "Score: " + str(self._game_engine.get_score())
        self.create_text(5, 5, text=msg, font=('Consolas', 14), anchor='nw', tags='score')

    def __x_to_screen(self, x):
        return Config.STEP_X * x + Config.DELTA_XY

    def __y_to_screen(self, y):
        return Config.HEIGHT - Config.STEP_Y - Config.STEP_Y * y + Config.DELTA_XY


class MainForm(tk.Tk):
    def __init__(self, game_engine: GameEngine):
        super().__init__()
        self._game_engine = game_engine

        f_top = tk.Frame(padx=5, pady=2)
        f_top.pack(fill='x')

        f_bottom = tk.Frame()
        f_bottom.pack()

        label_restart = tk.Label(f_top, text='Press <R> to restart game', font=('Consolas', 12))
        label_restart.pack(side='left')

        gb = GameBoard(f_bottom, game_engine)
        gb.pack()

        self.wm_title("Snake Super Game")
        self.eval('tk::PlaceWindow . center')
        self.resizable(False, False)
