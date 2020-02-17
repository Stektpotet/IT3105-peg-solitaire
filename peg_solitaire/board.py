from abc import ABC, abstractmethod
import numpy as np
import numpy.ma as ma

class Board(ABC):
    pegs: np.ndarray
    size: int
    hole_count: int

    @property
    def shape(self): return self.pegs.shape

    @property
    def peg_count(self): return np.sum(self.pegs)

    @property
    def full_count(self) -> int:
        return self.hole_count - 1

    @abstractmethod
    def _count_holes(self) -> int: pass

    # TODO: Look at what is most useful; flat indices or 2d indices - which is more used? Then unify...
    @abstractmethod
    def valid_action(self, peg_flat_index: int, move_index: int): pass

    @abstractmethod
    def apply_action(self, peg_flat_index: int, move_index: int): pass

    def valid_moves(self, peg_index2d: (int, int)):
        move_indices = []
        # We don't need to check if the skippable is there,
        # We need to check if the one past the skippable is there, hence the ' *2 '
        for move_index, move in enumerate(self.moves):
            y_axis = peg_index2d[0] + move[0] * 2
            x_axis = peg_index2d[1] + move[1] * 2
            if 0 <= y_axis < self.shape[0] and 0 <= x_axis < self.shape[1]:
                if not ma.is_masked(self.pegs[y_axis, x_axis]):
                    move_indices.append(move_index)

        return move_indices

    def __init__(self, size: int):
        """
        Moves:
                    0---1
                    | \ | \
                    5---X---2
                      \ | \ |
                        4---3
        """
        self.size = size
        self.hole_count = self._count_holes()
        self.moves = [(-1, -1), (-1, 0), (0, 1), (1, 1), (1, 0), (0, -1)]
        pass

    def __repr__(self):
        return f"<Board\nshape: {self.shape}\npegs: \n{self.pegs}\n>"

    def peg(self, flat_index: int):
        return self.pegs.flat[flat_index]

    def index_flat(self, pos: (int, int)):
        """
        Get the index of the flattened board when
        :param pos: x, y
        :return: flat index
        """
        return pos[1] + pos[0] * self.shape[1]

    def index_2d(self, i: int):
        """
        flat index to 2D index
        :param i: flat index
        :return: 2D tuple index
        """
        return int(i / self.shape[1]), i % self.shape[1]


class TriangleBoard(Board):

    def apply_action(self, peg_flat_index: int, move_index: int):
        # TODO: Implement
        pass

    def valid_action(self, peg_flat_index: int, move_index: int):
        # TODO: Implement
        return False
        pass

    def _count_holes(self) -> int:
        return len(self.indices[0])

    def __init__(self, size: int):
        self._unmasked_pegs = np.tri(size, dtype=int)
        self.pegs = ma.masked_array(self._unmasked_pegs, mask=np.tri(size, dtype=bool, k=-1).T, hard_mask=True)
        indices = np.tril_indices_from(self.pegs)
        self.indices = list(zip(*indices))

        self.flat_indices = [self.index_flat(i) for i in self.indices]
        self.pegs[int(self.shape[0] / 2), int(self.shape[1] / 2) - int(self.shape[1] / 4)] = 0
        Board.__init__(self, size)

    def __repr__(self):
        return f"<Board\nshape: {self.shape}\npegs: \n{self.pegs}\n>"


class DiamondBoard(Board):

    def _count_holes(self) -> int:
        """
        :return: The max number of pegs on the board, i.e. nr of holes - 1
        """
        return self.shape[0] * self.shape[1]

    def __init__(self, size: int):
        self.pegs = np.ones((size, size), dtype=int)
        self.pegs[int(self.shape[1] / 2), int(self.shape[0] / 2)] = 0
        Board.__init__(self, size)

    # NOTE: The following may be generalizable enough to put it into the superclass
    def _peg_line(self, peg_flat_index: int, move_index: int):
        peg_pos = self.index_2d(peg_flat_index)
        peg_skip_pos = (peg_pos[0] + self.moves[move_index][0], peg_pos[1] + self.moves[move_index][1])
        peg_jumper = (peg_pos[0] + self.moves[move_index][0] * 2, peg_pos[1] + self.moves[move_index][1] * 2)
        return peg_pos, peg_skip_pos, peg_jumper

    def valid_action(self, peg_flat_index: int, move_index: int):
        """
        Catch all cases of invalid actions before assuming it's a valid one
        :param peg_flat_index: the index of the peg hole we want filled
        :param move_index: the index of which direction to source a peg from
        :return:
        """
        if self.pegs.flat[peg_flat_index]:
            print(f"peg {peg_flat_index} at {self.index_2d(peg_flat_index)} is already filled!")
            return False

        if move_index < 0 or move_index >= len(self.moves):
            print(f"Illegal move - not defined in 'board.moves'")
            return False

        peg_pos, peg_skip_pos, peg_jumper = self._peg_line(peg_flat_index, move_index)

        if not self.pegs[peg_skip_pos]:
            print(f"Illegal move - cannot skip over {peg_skip_pos}, it's not filled!")
            return False
        try:
            if not self.pegs[peg_jumper]:
                print(f"Illegal move - cannot skip from {peg_jumper}, it's not filled!")
                return False
        except IndexError:
            print(f"WAT\t{peg_jumper} -> {peg_skip_pos} -> {peg_pos}")


        return True

    def apply_action(self, peg_flat_index: int, move_index: int):
        self.pegs.flat[peg_flat_index] = True  # Fill

        peg_pos, peg_skip_pos, peg_jumper = self._peg_line(peg_flat_index, move_index)

        self.pegs[peg_pos] = True
        self.pegs[peg_skip_pos] = False
        self.pegs[peg_jumper] = False

