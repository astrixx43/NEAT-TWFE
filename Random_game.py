from TWFE import get_board
import random
import time


ButtonNames = [
    "Up",
    "Down",
    "Left",
    "Right"
]


board = get_board()


def random_pad():
    controller = {}

    for b in range(len(ButtonNames)):
        controller[str(ButtonNames[b])] = False

    controller[(random.choice(ButtonNames))] = True

    board.input_movements(controller)

    board.update()

while True:
    random_pad()
    time.sleep(1)
    # board.update_idletasks()
    board.update()
