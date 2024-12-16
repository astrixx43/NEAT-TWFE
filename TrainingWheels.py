import TWFE
import random

board = TWFE.get_board()

class Player:
    def __init__(self):
        self.score = 0
        self.strategies = []
        self.moves = []

    def random_move(self):
        self.moves.append(random.choice(["Up", "Down", "Right", "Left"]))
        self.make_move(self.moves[-1])

    def make_move(self, move):
        s1 = board.Score
        board.input_movements_list(move)
        s2 = board.Score
        s = s2 - s1
        if self.moves == "Left":
            s -= s2 // 10
        self.score = 0

    def test_moves(self):

        n = 3
        best_move = self._test_move(n)
        board.load_state(best_move[2])

    def _test_move(self, n) -> tuple[int, str, list[list[int]]]:
        save = board.save_state()
        if n == 0:
            self.make_move("Up")
            save_up = board.save_state()
            if board.game_status:
                move_up = (board.score() - 2, "Up", save_up)
            else:
                move_up = (-1, "Up", save_up)
            board.load_state(save)

            self.make_move("Right")
            save_right = board.save_state()
            if board.game_status:
                move_right = (board.score() - 2, "Right", save_right)
            else:
                move_right = (-1, "Right", save_right)
            board.load_state(save)

            self.make_move("Down")
            save_down = board.save_state()
            if board.game_status:
                move_down = (board.score() - 2, "Down", save_down)
            else:
                move_down = (-1, "Down", save_down)
            board.load_state(save)

            self.make_move("Left")
            save_left = board.save_state()
            if board.game_status:
                move_left = (board.score() - 2, "Left", save_left)
            else:
                move_left = (-1, "Left", save_left)
            board.load_state(save)

            best_move = max(move_up, move_right, move_left, move_down)
            return best_move

        if n == 1:
            self.make_move("Up")
            up = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Right")
            right = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Down")
            down = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Left")
            left = self._test_move(n - 1)
            board.load_state(save)

            best_move = max(up, right, down, left)
            current_move = (best_move[0], "Up", save)
            return current_move

        else:
            self.make_move("Up")
            up = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Right")
            right = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Down")
            down = self._test_move(n - 1)
            board.load_state(save)

            self.make_move("Left")
            left = self._test_move(n - 1)
            board.load_state(save)

            return max(up, right, down, left)

p1 = Player()

board.update()
save = board.save_state()
while True:
    while board.game_status:
        p1.test_moves()
        board.update()
    print(board.Score)
    board.load_state(save)
