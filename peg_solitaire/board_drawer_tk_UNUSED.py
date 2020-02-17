import tkinter as tk
import numpy as np
from peg_solitaire.board import Board


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)


def _line_from_to(self, p0, p1, **kwargs):
    return self.create_line(p0[0], p0[1], p1[0], p1[1], **kwargs)


tk.Canvas.create_circle = _create_circle
tk.Canvas.line_from_to = _line_from_to


def scale(mat, x, y):
    return mat.dot(np.array([(x, 0, 0), (0, y, 0), (0, 0, 1)]))


def translate(mat, x, y):
    return mat.dot(np.array([(1, 0, 0), (0, 1, 0), (x, y, 1)]))


def rotate(mat, theta):
    c, s = np.cos(theta), np.sin(theta)
    return mat.dot(np.array([(c, -s, 0), (s, c, 0), (0, 0, 1)]))

class BoardDrawer:
    canvas: tk.Canvas

    def __init__(self, canvas: tk.Canvas, peg_scale=0.5, board_scale=1):
        self.canvas = canvas
        self.transform = np.identity(3)
        self.board_scale = board_scale
        self.peg_scale = peg_scale

    def draw(self, board: Board):

        view_transform = translate(np.identity(3), board.shape[1]*0.5, board.shape[0]*0.5)
        view_transform = scale(view_transform, self.board_scale/board.shape[1], self.board_scale/board.shape[0])

        for y in range(board.shape[0]):
            for x in range(board.shape[1]):

                o = np.array([x, y, 1], dtype=np.float64)
                p = o.dot(self.transform.dot(view_transform))

                fill = "black" if board.pegs[x, y] else "white"

                n = np.array([(max(x - 1, 0), max(y - 1, 0), 1),  # left
                              (x, max(y - 1, 0), 1),  # up to the left
                              (max(x - 1, 0), y, 1)])  # down to the left

                for i in range(len(n)):
                    self.canvas.line_from_to(p, np.array(n[i]).dot(self.transform.dot(self.view)))

                # TODO: hold an array of these instead of access through tag?
                self.canvas.create_circle(p[0], p[1], 90 * self.peg_size / self.size, fill=fill,
                                          tags=("peg", f'({x}, {y})'))
        self.canvas.tag_raise("peg")