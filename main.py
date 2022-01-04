from domain import GameEngine
from gui import MainForm


def main():
    game = GameEngine(20, 20, 0.1)
    form = MainForm(game)
    game.start()
    form.mainloop()


if __name__ == "__main__":
    main()
