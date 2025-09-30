import tkinter as tk
from itertools import cycle
from tkinter import font
from typing import NamedTuple
import random
import os.path


class Move(NamedTuple):
    row: int
    col: int
    label: str = ""


def list_setup():
    file = open("Leaderbaord.txt", "r")
    file_text = file.read()
    file.close()
    file_text = file_text[1:-1]
    file_list = []
    acc = 0
    if len(file_text) == 0:
        return []
    file_text = file_text.replace("'", "")
    for i in range(len(file_text)):
        sub = []
        rest = []
        if file_text[i] != "]":
            if file_text[i] == "[":
                p = file_text.find("]", i + 1)
                rest = file_text[i + 1: p]
                sub = rest.split(",")
                sub[0] = int(sub[0])
                sub[1] = int(sub[1])
                sub[3] = int(sub[3])
                acc += 1
                file_list.append(sub)
    leader_list = file_list
    leader_list.sort()
    return leader_list


if os.path.isfile("Leaderbaord.txt"):
    leader_list = list_setup()
else:
    file = open('Leaderbaord.txt', 'w')
    file.close()
    leader_list = list_setup()


def leaderboard_text(num_guesses: int) -> None:
    high_score = 0
    for j in range(len(leader_list)):
        if high_score < int(leader_list[j][1]):
            high_score = int(leader_list[j][1])

    if high_score < num_guesses:
        trys = "Your Score was: " + str(num_guesses) + " WOW!!! New High Score."
    else:
        trys = "Your Score was: " + str(num_guesses) + "."

    wining_text = "\n {:<10} \n {:<10}".format("Congrats!!!", trys)
    print(wining_text)
    name = input("name?")
    while "[" in name or "]" in name:
        name = input("No paranthesis, name?")
    leader_list.append([0, num_guesses, name, 0])
    leader_list.sort()
    file = open("Leaderbaord.txt", "w")
    file.write(str(leader_list))
    file.close()
    leader = 1
    text = "WINNERS...WINNERS...WINNERS... "
    print(text + "\n")
    print("{:}{:^10}{:>5}".format("Rank:", "Name,", "Score"))
    for i in range(len(leader_list) - 1, -1, -1):
        winner = "\t{:>5}, {:>8}".format(leader_list[i][2],
                                         leader_list[i][1])
        print(str(leader) + ": " + winner)
        leader += 1
    return None


def display_leaderboard():
    leaderboard = list_setup()
    leaderboard.sort()
    a = []
    for i in range(len(leaderboard)):
        for k in range(len(a) + 1):
            if (k != len(a) or len(a) == 0 or (
                    a[k][0] != leaderboard[i][0] and
                    a[k][3] != leaderboard[i][3])):
                text = "WINNERS IN THE CATAGORY OF RANGE " + str(
                    leaderboard[i][0])
                print("\n" + text + " AT DIFFICULTY " + str(
                    leaderboard[i][3] + 1) + "\n")
                print("{:}{:^10}{:>5}".format("Rank:", "Name,", "Score"))
                leader = 1
                for j in range(len(leaderboard) - 1, -1, -1):
                    if (leaderboard[i][0] == leaderboard[j][0] and
                            leaderboard[i][3] == leaderboard[j][3]):
                        winner = "\t{:>5}, {:>8}".format(leader_list[j][2],
                                                         leader_list[j][1])
                        print(str(leader) + ": " + winner)
                        a.append(leaderboard[i])
                        leader += 1
            break


class TWFENoUI:

    def __init__(self, DIM=4):
        self.Score = 0
        self.game_status = True
        self.mergeCount = 0
        self.movesMade = 0
        self.DIM = DIM
        self._board = []
        for i in range(DIM):
            self._board.append([])
            for j in range(DIM):
                self._board[i].append(0)
        self.new_btn()

    def _end(self):
        self._Game_Over = True
        # self._reset()

    def is_game_over(self):
        return self._Game_Over

    def new_btn(self) -> bool:
        pos = (random.randint(0, 3), random.randint(0, 3))
        i = 0
        while self.is_occupied(pos) and i != 500:
            row, col = random.randint(0, 3), random.randint(0, 3)
            pos = (row, col)
            i += 1

        if i < 500:
            self._set_pos_value(pos, random.choice([2, 4]))
            return True
        return False

    def _reset(self):
        self._Game_Over = False
        self.mergeCount = 0
        self.movesMade = 0
        for row in range(4):
            for col in range(4):
                self._set_pos_value((row, col), 0)

        self.new_btn()
        # leaderboard_text(self.score())

    def down(self):
        """Handle a player's move."""
        if self.game_over():
            initial_moves_made = self.movesMade

            for i in range(self.DIM):
                self._down_button()

            if initial_moves_made < self.movesMade:
                self.new_btn()

            if not self.game_over():
                self._end()

        else:
            self._end()

    def right(self):
        if self.game_over():
            initial_moves_made = self.movesMade

            for i in range(self.DIM):
                self._right_button()

            if initial_moves_made < self.movesMade:
                self.new_btn()

            if not self.game_over():
                self._end()

        else:
            self._end()

    def left(self):
        if self.game_over():
            initial_moves_made = self.movesMade

            for i in range(self.DIM):
                self._left_button()

            if initial_moves_made < self.movesMade:
                self.new_btn()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def up(self):
        """Handle a player's move."""
        if self.game_over():
            initial_moves_made = self.movesMade

            for i in range(self.DIM):
                self._up_button()

            if initial_moves_made < self.movesMade:
                self.new_btn()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def score(self):
        score = 0
        for i in range(self.DIM):
            for j in range(self.DIM):
                pos = (i, j)
                score += self.pos_to_value(pos)

        self.Score = score
        return score

    def getEmptyTileSize(self):
        num = self.DIM ** 2
        for row in range(self.DIM):
            for col in range(self.DIM):
                if not self.is_occupied((row, col)):
                    num -= 1
        return num

    def getLargestTile(self):
        largest = 0
        for row in range(self.DIM):
            for col in range(self.DIM):
                num = self.pos_to_value((row, col))
                if largest < num:
                    largest = num

        return largest

    def _up_button(self):
        for i in range(self.DIM):
            if i == 0:
                for row in range(self.DIM):
                    for col in range(self.DIM - 1 , -1, -1):
                        pos = (row, col)
                        adj_pos = (pos[0] - 1, col)
                        while pos[0] > -1:
                            self.collision(pos, adj_pos)
                            pos = (pos[0] - 1, col)
                            adj_pos = (pos[0] - 1, col)
        self._post_collison()

    def _down_button(self):
        for i in range(self.DIM):
            if i == 0:
                for row in range(self.DIM - 1, -1, -1):
                    for col in range(self.DIM - 1, -1, -1):
                        pos = (row, col)
                        adj_pos = (pos[0] + 1, col)
                        while pos[0] < 4:
                            self.collision(pos, adj_pos)
                            pos = (pos[0] + 1, col)
                            adj_pos = (pos[0] + 1, col)
        self._post_collison()

    def _right_button(self):
        for i in range(self.DIM):
            if i == 0:
                for col in range(self.DIM - 1, -1, -1):
                    for row in range(self.DIM - 1, -1, -1):

                        pos = (row, col)
                        adj_pos = (pos[0], col + 1)
                        while pos[1] < 4:
                            self.collision(pos, adj_pos)
                            pos = (row, pos[1] + 1)
                            adj_pos = (row, pos[1] + 1)

        self._post_collison()

    def _left_button(self):
        for i in range(self.DIM):
            if i == 0:
                for col in range(self.DIM):
                    for row in range(self.DIM - 1, -1, -1):
                        pos = (row, col)
                        adj_pos = (row, col - 1)
                        while pos[1] > -1:
                            self.collision(pos, adj_pos)
                            pos = (row, pos[1] - 1)
                            adj_pos = (row, pos[1] - 1)
        self._post_collison()

    def _post_collison(self):
        pass

    def is_valid_position(self, pos: tuple[int,int]):
        return 0 <= pos[0] < self.DIM and 0 <= pos[1] < self.DIM

    def pos_to_value(self, pos: tuple[int,int]):
        if self.is_valid_position(pos):
            return self._board[pos[0]][pos[1]]
        return -1

    def is_position_empty(self, pos: tuple[int,int]) -> bool:
        return self.is_valid_position(pos) and self._board[pos[0]][pos[1]] == 0

    def _set_pos_value(self, pos: tuple[int,int], value: int):
        if self.is_valid_position(pos):
            self._board[pos[0]][pos[1]] = value

    def collision(self, btn_pos_1, btn_pos_2):

        num1 = self.pos_to_value(btn_pos_1)
        num2 = self.pos_to_value(btn_pos_2)

        if num1 > -1 and num2 > -1 and (num1 == num2 or num1 == 0 or num2 == 0):

            if num1 == num2 and num1 != 0:
                self.mergeCount += 1

            self._set_pos_value(btn_pos_1, 0)
            self._set_pos_value(btn_pos_2, num1 + num2)
            # if not (num1 == 0 and num2 == 0) or (num1 != 0 and num2 == 0):
            self.movesMade += 1

            return 1

        return 0

    def is_occupied(self, pos, dirc=""):
        if not self.is_valid_position(pos):
            return True

        if dirc == "up":
            if pos[0] <= 0:
                return True
            new_pos = (pos[0] - 1, pos[1])
            return not self.is_position_empty(new_pos)

        elif dirc == "down":
            if pos[0] >= self.DIM - 1:
                return True
            new_pos = (pos[0] + 1, pos[1])
            return not self.is_position_empty(new_pos)

        elif dirc == "left":
            if pos[1] <= 0:
                return True
            new_pos = (pos[0], pos[1] - 1)
            return not self.is_position_empty(new_pos)

        elif dirc == "right":
            if pos[1] >= self.DIM - 1:
                return True
            new_pos = (pos[0], pos[1] + 1)

            return not self.is_position_empty(new_pos)


        return not self.is_position_empty(pos)

    def getMergeCount(self):
        return self.mergeCount

    def getNumMoves(self):
        return self.movesMade

    def game_over(self):
        for row in range(4):
            for col in range(4):
                pos = (row, col)
                num1 = self.pos_to_value(pos)

                if self.is_position_empty(pos):
                    return True

                if row < 3:
                    num2 = self.pos_to_value((row + 1, col))
                    if num1 == num2:
                        return True

                if row > 0:
                    num2 = self.pos_to_value((row - 1, col))
                    if num2 == num1:
                        return True

                if col < 3:
                    num2 = self.pos_to_value((row, col + 1))
                    if num2 == num1:
                        return True

                if col < 0:
                    num2 = self.pos_to_value((row, col - 1))
                    if num2 == num1:
                        return True

        return False

    def input_movements(self, joypad: dict):
        for i in joypad:
            if joypad[i]:

                if i == "Up":
                    self.up()
                elif i == "Down":
                    self.down()
                elif i == "Right":
                    self.right()
                elif i == "Left":
                    self.left()
        # print(joypad)

    def input_movements_list(self, i):
        if i == "Up":
            self.up()
        elif i == "Down":
            self.down()
        elif i == "Right":
            self.right()
        elif i == "Left":
            self.left()
        # print(joypad)

        # for move in joypad

    def __str__(self) -> str:
        string = ""
        for row in range(self.DIM):
            string += "\n ["
            for col in range(self.DIM):
                string += str(self.pos_to_value((row, col)))+ ", "

            string += "]"
        string += " \n"

        return string

    # def save_state(self) -> list[list[int]]:
    #     lst = []
    #     for i in range(4):
    #         new_row = []
    #         for j in range(4):
    #             new_row.append(self.btn_value((i, j)))
    #         lst.append(new_row)
    #     return lst
    #
    # def load_state(self, lst: list[list[int]]):
    #     score = 0
    #     for i in range(len(lst)):
    #         for j in range(len(lst[i])):
    #             score += lst[i][j]
    #             self.set_btn_value((i, j), lst[i][j])
    #     self.Score = score
    #     self.game_status = True
    #     self._update_display(self.score())
