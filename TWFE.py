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


class TFE(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("2048 THE GAME")
        self._cells = {}
        a = 0
        self.tiles = {}
        self._create_board_display()
        self._create_board_grid(a)
        self.colors = ["", "red", "green", "blue", "blueviolet", "brick",
                       "brown1",
                       "cadetblue1", "cadmiumorange", "indianred1", "lawngreen",
                       "lightcoral", "manganeseblue", "maroon1", "orange",
                       "palevioletred3",
                       "peacock", "gold"]
        self.Score = 0
        self.game_status = True
        self.mergeCount = 0
        self.movesMade = 0
        grid_frame = tk.Frame(master=self)
        grid_frame.pack()

    def _create_board_display(self):
        display_frame = tk.Frame(master=self)
        display_frame.pack(fill=tk.X)
        self.display = tk.Label(
            master=display_frame,
            text="2048",
            font=font.Font(size=28, weight="bold"),
        )
        self.display.pack()

    def _update_display(self, msg, color="black"):
        self.display["text"] = msg
        self.display["fg"] = color

    def _create_board_grid(self, a):
        grid_frame = tk.Frame(master=self)
        grid_frame.pack()
        for row in range(4):
            self.rowconfigure(row, weight=1, minsize=50)
            self.columnconfigure(row, weight=1, minsize=75)
            for col in range(4):
                button = tk.Button(
                    master=grid_frame,
                    text="",
                    font=font.Font(size=36, weight="bold"),
                    fg="black",
                    width=3,
                    height=1,
                    highlightbackground="lightblue",
                )

                self._cells[button] = (row, col)
                self.tiles[(row, col)] = button
                button.grid(row=row, column=col, padx=0, pady=0, sticky="nsew")

                # num = random.randint(0, 16)
                #
                # if num == rnum:
                #     row, col = self._cells[button]
                #     self.rand_but(button)
                #     rnum = -1
                #     a = 1
                # elif row == 3 and a == 0 and col == 3:
                #     rnum = -1
                #     a = 1
                #     row, col = self._cells[button]
                #     self.rand_but(button)

                button.bind("<Key-w>", self.up)
                button.bind("<Key-s>", self.down)
                button.bind("<Key-d>", self.right)
                button.bind("<Key-a>", self.left)
                button.bind("<Up>", self.up)
                button.bind("<Down>", self.down)
                button.bind("<Right>", self.right)
                button.bind("<Left>", self.left)
        self.new_btn()

    def _end(self):
        self._update_display("Game Over")
        self._Game_Over = True
        # self._reset()

    def is_game_over(self):
        return self._Game_Over

    def new_btn(self):
        pos = (random.randint(0, 3), random.randint(0, 3))
        i = 0
        while self.is_ocupied(pos, "") and i != 50:
            row, col = random.randint(0, 3), random.randint(0, 3)
            pos = (row, col)
            i += 1
        if i < 50:
            btn = self.pos_to_btn(pos)
            btn['text'] = str(random.choice([2, 4]))
            self._update_display(self.score())

    def _reset(self):
        self._Game_Over = False
        self.mergeCount = 0
        self.movesMade = 0
        for row in range(4):
            for col in range(4):
                pos = (row, col)
                btn = self.pos_to_btn(pos)
                btn['text'] = ""
        self.new_btn()
        # leaderboard_text(self.score())

    def up(self, event):
        """Handle a player's move."""
        if self.game_over():

            self._up_button()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def down(self, event):
        """Handle a player's move."""
        if self.game_over():

            self._down_button()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def left(self, event):
        if self.game_over():

            self._left_button()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def right(self, event):
        if self.game_over():

            self._right_button()

            if not self.game_over():
                self._end()
        else:
            self._end()

    def score(self):
        score = 0
        for i in range(4):
            for j in range(4):
                pos = (i, j)
                score += self.btn_value(pos)

        self.Score = score
        return score

    def getEmptyTileSize(self):
        num = 16
        for row in range(4):
            for col in range(4):
                if not self.is_ocupied((row, col)):
                    num -= 1
        return num

    def getLargestTile(self):
        largest = 0
        for row in range(4):
            for col in range(4):
                num = self.btn_value((row, col))
                if largest < num:
                    largest = num

        return largest

    def pos_to_btn(self, pos: tuple[int, int]):
        return self.tiles[pos]

    def _up_button(self):
        a = 0
        for i in range(4):
            for row in range(4):
                for col in range(3, -1, -1):
                    pos = (row, col)
                    adj_pos = (pos[0] - 1, col)
                    while pos[0] > -1:
                        a += self.collision(pos, adj_pos)
                        pos = (pos[0] - 1, col)
                        adj_pos = (pos[0] - 1, col)

        self._post_collison(a)

    def _down_button(self):
        a = 0
        for i in range(4):
            for row in range(3, -1, -1):
                for col in range(3, -1, -1):
                    pos = (row, col)
                    adj_pos = (pos[0] + 1, col)
                    while pos[0] < 4:
                        a += self.collision(pos, adj_pos)
                        pos = (pos[0] + 1, col)
                        adj_pos = (pos[0] + 1, col)

        self._post_collison(a)

    def _right_button(self):
        a = 0
        for i in range(4):
            for col in range(3, -1, -1):
                for row in range(3, -1, -1):
                    pos = (row, col)
                    adj_pos = (pos[0], col + 1)
                    while pos[1] < 4:
                        a += self.collision(pos, adj_pos)
                        pos = (row, pos[1] + 1)
                        adj_pos = (row, pos[1] + 1)
        self._post_collison(a)

    def _left_button(self):
        a = 0
        for i in range(4):
            for col in range(4):
                for row in range(3, -1, -1):
                    pos = (row, col)
                    adj_pos = (row, col - 1)
                    while pos[1] > -1:
                        a += self.collision(pos, adj_pos)
                        pos = (row, pos[1] - 1)
                        adj_pos = (row, pos[1] - 1)

        self._post_collison(a)

    def _post_collison(self, a):
        if a > 0:
            self.movesMade += 1
            self.new_btn()

        for row in range(4):
            for col in range(4):
                btn = self.pos_to_btn((row, col))
                if btn["text"] == " ":
                    btn["text"] = ""
                    btn.config(fg="black")

    def collision(self, btn_1, btn_2):
        btn_1 = self.pos_to_btn(btn_1)
        if -1 < btn_2[0] < 4 and -1 < btn_2[1] < 4:
            btn_2 = self.pos_to_btn(btn_2)

            em = [" ", ""]

            if (btn_2['text'] not in em and btn_1['text'] not in em and
                    btn_1['text'] == btn_2['text']):

                num1 = int(btn_1['text'])
                num2 = int(btn_2['text'])
                btn_1.config(text=" ")
                btn_2.config(text=str(num1 + num2))
                self.mergeCount += 1

                return 1

            elif btn_2['text'] in em and btn_1['text'] not in em:
                num1 = int(btn_1['text'])
                btn_1.config(text="")
                btn_2.config(text=str(num1))
                btn_2.config(fg="black")
                return 1

            elif btn_2['text'] == "" and btn_1['text'] == " ":
                btn_1.config(text="")
                btn_2.config(text=" ")
                btn_2.config(fg="black")

            else:
                btn_2.config(fg="black")
                btn_1.config(fg="black")

        elif btn_1['fg'] != "black":
            btn_1['fg'] = "black"

        return 0

    def is_ocupied(self, pos, dir=""):
        em = ["", " "]
        if dir == "up":
            if pos[0] <= 0:
                return True
            btn = self.pos_to_btn((pos[0] - 1, pos[1]))

            return btn['text'] not in em

        elif dir == "down":
            if pos[0] >= 3:
                return True
            btn = self.pos_to_btn((pos[0] + 1, pos[1]))
            return btn['text'] not in em

        elif dir == "left":
            if pos[1] <= 0:
                return True
            btn = self.pos_to_btn((pos[0], pos[1] - 1))
            return btn['text'] not in em

        elif dir == "right":
            if pos[1] >= 3:
                return True
            btn = self.pos_to_btn((pos[0], pos[1] + 1))
            return btn['text'] not in em

        elif 0 <= pos[0] <= 3 and 0 <= pos[1] <= 3:
            btn = self.pos_to_btn((pos[0], pos[1]))
            return btn['text'] not in em
        return True

    def getMergeCount(self):
        return self.mergeCount

    def getNumMoves(self):
        return self.movesMade

    def game_over(self):
        for row in range(4):
            for col in range(4):
                pos = (row, col)
                btn = self.pos_to_btn(pos)
                adj_btn_1 = ""
                adj_btn_2 = ""
                adj_btn_3 = ""
                adj_btn_4 = ""
                em = ["", " "]
                if btn['text'] == "" or btn['text'] == " ":
                    return True

                if row < 3:
                    adj_btn_2 = self.pos_to_btn((row + 1, col))
                    if (adj_btn_2['text'] in em or
                            btn['text'] == adj_btn_2['text']):
                        return True

                if row > 0:
                    adj_btn_1 = self.pos_to_btn((row - 1, col))
                    if (adj_btn_1['text'] in em or
                            btn['text'] == adj_btn_1['text']):
                        return True

                if col < 3:
                    adj_btn_3 = self.pos_to_btn((row, col + 1))
                    if (adj_btn_3['text'] in em or
                            btn['text'] == adj_btn_3['text']):
                        return True

                if col < 0:
                    adj_btn_4 = self.pos_to_btn((row, col - 1))
                    if (adj_btn_4['text'] == "" or
                            btn['text'] == adj_btn_4['text']):
                        return True

        return False

    def input_movements(self, joypad: dict):
        for i in joypad:
            if joypad[i]:
                self.movesMade += 1
                if i == "Up":
                    self.up("Up")
                elif i == "Down":
                    self.down("Down")
                elif i == "Right":
                    self.right("Right")
                elif i == "Left":
                    self.left("Left")
        # print(joypad)

    def input_movements_list(self, i):
        if i == "Up":
            self.up("Up")
        elif i == "Down":
            self.down("Down")
        elif i == "Right":
            self.right("Right")
        elif i == "Left":
            self.left("Left")
        # print(joypad)

        # for move in joypad

    def btn_value(self, pos: tuple[int, int]):
        btn = self.pos_to_btn(pos)
        if btn['text'] in [" ", ""]:
            return 0
        return int(btn['text'])

    def set_btn_value(self, pos: tuple[int, int], value: int):
        btn = self.pos_to_btn(pos)
        if value == 0:
            btn['text'] = ""
        else:
            btn['text'] = str(value)

    def save_state(self) -> list[list[int]]:
        lst = []
        for i in range(4):
            new_row = []
            for j in range(4):
                new_row.append(self.btn_value((i, j)))
            lst.append(new_row)
        return lst

    def load_state(self, lst: list[list[int]]):
        score = 0
        for i in range(len(lst)):
            for j in range(len(lst[i])):
                score += lst[i][j]
                self.set_btn_value((i, j), lst[i][j])
        self.Score = score
        self.game_status = True
        self._update_display(self.score())


def main():
    """Create the game's board and run its main loop."""
    board = TFE()
    board.mainloop()

#
# board = TFE()
# board.mainloop()


def get_board():
    board = TFE()
    return board

# if __name__ == "__main__":
#     # main()

#
#
#
# board = get_board()
# while True:
#     board.update()

# Leaderboard
# Instruction Manual
# Colored blocks (Up to 17 cuz 2**17)
# Some collison issues need to be adderes(Bug fix them. This implies bugtestin):
#
