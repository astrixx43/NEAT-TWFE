class TWFE():

    def __init__(self):
        self._cells = {}
        for row in range(4):
            for col in range(4):
                self._cells[(row,col)] = 0

        self.Score = 0
        self.game_status = True
        self.mergeCount = 0
        self.movesMade = 0

    # def play(self):


    def _end(self):
        self._update_display("Game Over")
        self._Game_Over = True
        # self._reset()

    def is_game_over(self):
        return self._Game_Over

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

    def down(self, event):
        """Handle a player's move."""
        if self.game_over():

            for i in range(4):
                self._down_button()
            self.new_btn()
            if not self.game_over():
                self._end()
        else:
            self._end()

    def right(self, event):
        if self.game_over():

            for i in range(4):
                self._right_button()
            self.new_btn()
            if not self.game_over():
                self._end()
        else:
            self._end()

    def left(self, event):
        if self.game_over():
            for i in range(4):
                self._left_button()
            self.new_btn()
            if not self.game_over():
                self._end()
        else:
            self._end()

    def up(self, event):
        """Handle a player's move."""
        if self.game_over():
            for i in range(4):
                self._up_button()
            self.new_btn()
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

    def pos_to_btn(self, pos: tuple[int,int]):
        return self.tiles[pos]

    def _up_button(self):
        for i in range(4):
            if i == 0:
                for row in range(4):
                    for col in range(3, -1, -1):
                        a = 0
                        pos = (row, col)
                        adj_pos = (pos[0] - 1, col)
                        while pos[0] > -1:
                            self.collision(pos, adj_pos, a)
                            pos = (pos[0] - 1, col)
                            adj_pos = (pos[0] - 1, col)
        self._post_collison()

    def _down_button(self):
        for i in range(4):
            if i == 0:
                for row in range(3, -1, -1):
                    for col in range(3, -1, -1):
                        a = 0
                        b = 0
                        pos = (row, col)
                        adj_pos = (pos[0] + 1, col)
                        while pos[0] < 4 and a == 0 and b == 0:
                            self.collision(pos, adj_pos, a)
                            pos = (pos[0] + 1, col)
                            adj_pos = (pos[0] + 1, col)
        self._post_collison()

    def _right_button(self):
        for i in range(4):
            if i == 0:
                for col in range(3, -1, -1):
                    for row in range(3, -1, -1):
                        b = 0
                        a = 0
                        pos = (row, col)
                        adj_pos = (pos[0], col + 1)
                        while pos[1] < 4 and a == 0 and b == 0:
                            self.collision(pos, adj_pos, a)
                            pos = (row, pos[1] + 1)
                            adj_pos = (row, pos[1] + 1)
        self._post_collison()

    def _left_button(self):
        for i in range(4):
            if i == 0:
                for col in range(4):
                    for row in range(3, -1, -1):
                        a = 0
                        b = 0
                        pos = (row, col)
                        adj_pos = (row, col - 1)
                        while pos[1] > -1 and a == 0 and b == 0:
                            self.collision(pos, adj_pos, a)
                            pos = (row, pos[1] - 1)
                            adj_pos = (row, pos[1] - 1)
        self._post_collison()

    def _post_collison(self):
        for row in range(4):
            for col in range(4):
                btn = self.pos_to_btn((row, col))
                if btn["text"] == " ":
                    btn["text"] = ""
                    btn.config(fg="black")

    def collision(self, btn_1, btn_2, a):
        btn_1 = self.pos_to_btn(btn_1)
        if -1 < btn_2[0] < 4 and -1 < btn_2[1] < 4:
            btn_2 = self.pos_to_btn(btn_2)
            em = [" ", ""]
            if btn_2['text'] not in em and btn_1['text'] not in em and btn_1[
                'text'] == btn_2['text'] and btn_2["fg"] != "red" and btn_1[
                "fg"] != "red":
                num1 = int(btn_1['text'])
                num2 = int(btn_2['text'])
                btn_1.config(text=" ")
                btn_2.config(text=str(num1 + num2))
                self.mergeCount += 1
                btn_2.config(fg="red")
                return 1
            elif btn_2['text'] == "" and btn_1['text'] not in em:
                num1 = int(btn_1['text'])
                btn_1.config(text="")
                btn_2.config(text=str(num1))
                btn_2.config(fg="black")
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