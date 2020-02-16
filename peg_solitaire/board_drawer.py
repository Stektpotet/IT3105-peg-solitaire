import numpy as np
from peg_solitaire.board import *
from skimage.draw import (line, circle, circle_perimeter)

def scale(mat, x, y):
    return mat.dot(np.array([(x, 0, 0), (0, y, 0), (0, 0, 1)]))


def translate(mat, x, y):
    return mat.dot(np.array([(1, 0, 0), (0, 1, 0), (x, y, 1)]))


def rotate(mat, theta):
    c, s = np.cos(theta), np.sin(theta)
    return mat.dot(np.array([(c, -s, 0), (s, c, 0), (0, 0, 1)]))


def shear_horizontal(mat, k):
    return mat.dot(np.array([(1, k, 0), (0, 1, 0), (0, 0, 1)]))


class BoardDrawer:
    _image: np.ndarray
    _view: np.ndarray
    peg_scale: int

    def _draw_circle_filled(self, x: int, y: int, fill=None):
        # circle fill
        if not fill:
            fill = self.fill_color
        rr, cc = circle(x, y, self.peg_scale, shape=self._image.shape)
        self._image[rr, cc] = fill
        self._draw_circle(x, y)

    def _draw_circle(self, x: int, y: int):
        # circle outline
        rr, cc = circle_perimeter(x, y, self.peg_scale, shape=self._image.shape)
        self._image[rr, cc, :] = self.color

    def _draw_line(self, p_x: int, p_y: int, q_x: int, q_y: int):
        """
        Draw line from p to q (screen space)
        :param p: line start
        :param q: line end
        """
        # circle outline
        rr, cc = line(p_x, p_y, q_x, q_y)
        self._image[rr, cc, :] = self.color

    @property
    def size(self) -> int: return self._image.shape[0]

    def __init__(self, resolution: int,
                 peg_scale: float = 0.25, board_scale: float = 1,
                 outline_color=(255, 255, 255), fill_color=(127, 0, 0)):

        self._image = np.zeros((resolution, resolution, 3), dtype=np.ubyte)  # Make a black image

        self.board_scale = board_scale
        self.peg_scale = int(peg_scale * 0.0625 * resolution)
        self._view = self._image.view()

        self.color = outline_color
        self.fill_color = fill_color

    @staticmethod
    def board_type_transform(mat: np.ndarray, board: Board):
        if type(board) is TriangleBoard:
            x = shear_horizontal(mat, -0.5)
            x = translate(x, 1, (board.size-1)*0.5+1)
            return x

        elif type(board) is DiamondBoard:
            x = translate(mat, -(board.size-1)*0.5, -(board.size-1)*0.5) # move the board so its center is in (0, 0)
            x = rotate(x, -np.radians(45))  # rotate the board around  (0, 0)
            x = scale(x, 0.707, 0.707)  # scale by 1/sqrt(2)
            x = scale(x, 1, .6)  # squish, make hexagons-like sections
            x = translate(x, (board.size+1)*0.5, (board.size+1)*0.5)  # move to center
            return x
        else:
            print("Unknown board type!")
            return mat

    def _draw_lines(self, board: Board, view, transform):
        pass

    def _draw_pegs(self, board: Board, view, transform):
        pass

    def draw(self, board: Board):


        # Center the board around the origin (0, 0)
        transform = self.board_type_transform(np.identity(3), board)

        # Apply the appropriate transformation for the board
        # transform = self.board_type_transform(board, transform)

        view_transform = scale(np.identity(3),
                               self.size/(board.size+1),
                               self.size/(board.size+1))



        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if ma.is_masked(board.pegs[x, y]):
                    continue

                o = np.array([x, y, 1], dtype=int)  # We need a 2D homogeneous vector (i.e. x,y,z) for matrix ops
                p = o.dot(transform.dot(view_transform))  # Transform point to screen space

                # TODO use board.moves - or even better, some 'valid move' array
                n = np.array([
                    [max(x - 1, 0), max(y - 1, 0), 1],  # left
                    [x, max(y - 1, 0), 1],  # up to the left
                    [min(x + 1, board.size-1), y, 1]
                    ])  # down to the left

                p_x = int(p[0])
                p_y = int(p[1])

                for i in range(len(n)):
                    q_x, q_y, *_ = n[i].dot(transform.dot(view_transform))
                    self._draw_line(p_x, p_y, int(q_x), int(q_y))


        for y in range(board.shape[0]):
            for x in range(board.shape[1]):

                if ma.is_masked(board.pegs[x, y]):
                    continue
                o = np.array([x, y, 1], dtype=int)  # We need a 2D homogeneous vector (i.e. x,y,z) for matrix ops
                p = o.dot(transform.dot(view_transform))  # Transform point to screen space

                p_x = int(p[0])
                p_y = int(p[1])

                if board.pegs[x, y]:
                    self._draw_circle_filled(p_x, p_y)
                else:
                    self._draw_circle_filled(p_x, p_y, fill=(255, 255, 255))

        return self._view
