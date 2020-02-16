from abc import ABC, abstractmethod
import numpy as np
import numpy.ma as ma
import networkx as nx

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

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        return graph

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


class TriangleBoard(Board):

    def _count_holes(self) -> int:
        return len(self.indices[0])

    def __init__(self, size: int):
        self._unmasked_pegs = np.tri(size, dtype=int)
        self.pegs = ma.masked_array(self._unmasked_pegs, mask=np.tri(size, dtype=bool, k=-1).T, hard_mask=True)
        indices = np.tril_indices_from(self.pegs)
        self.indices = list(zip(indices[0], indices[1]))

        self.flat_indices = [self.index_flat(i) for i in self.indices]
        self.pegs[int(self.shape[0] / 2), int(self.shape[1] / 2) - int(self.shape[1] / 4)] = 0
        Board.__init__(self, size)

    def __repr__(self):
        return f"<Board\nshape: {self.shape}\npegs: \n{self.pegs}\n>"

    def index_flat(self, pos: (int, int)):
        """
        Get the index of the flattened board when
        :param pos: x, y
        :return: flat index
        """
        return pos[0] + pos[1] * self.shape[1]

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        print(self.indices)
        x = [(i[0]+m[0], i[1]+m[1]) for m in self.moves for i in self.indices]
        print(x)
        graph.add_edges_from(x)
        return graph

class DiamondBoard(Board):

    @property
    def height(self): return self.pegs.shape[0]

    @property
    def width(self): return self.pegs.shape[1]

    def _count_holes(self) -> int:
        """
        :return: The max number of pegs on the board, i.e. nr of holes - 1
        """
        return self.height * self.width

    def __init__(self, size: int):
        self.pegs = np.ones((size, size), dtype=int)
        self.pegs[int(self.shape[1] / 2), int(self.shape[0] / 2)] = 0
        Board.__init__(self, size)

    def index_flat(self, pos: (int, int)):
        """
        Get the index of the flattened board when
        :param pos: x, y
        :return: flat index
        """
        return pos[0] + pos[1] * self.width

    def index_2d(self, i: int):
        """
        flat index to 2D index
        :param i: flat index
        :return: 2D tuple index
        """
        return int(i / self.shape[1]), i % self.shape[1]

    def graph(self) -> nx.Graph:
        pass
