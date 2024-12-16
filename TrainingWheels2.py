import TWFE
import random


class Player:
    def __init__(self):
        self.score = 0
        self.strategies = []
        self.moves = []

    def random_move(self):
        self.moves.append(random.choice(["Up", "Down", "Right", "Left"]))

    def make_move(self):
        s1 = board.Score
        board.input_movements_list(self.moves[-1])
        s2 = board.Score

board = TWFE.get_board()

p1 = Player()

