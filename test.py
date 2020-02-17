import matplotlib.pyplot as plt
import argparse as arg
import numpy as np
import threading
import numpy.ma as ma
# A runnable file to test out various things

def index_flat(arr: np.ndarray, pos: (int, int)):
    """
    Get the index of the flattened board when
    :param pos: x, y
    :return: flat index
    """
    return pos[1] + pos[0] * arr.shape[1]

def index_2d(arr, i: int):
        """
        flat index to 2D index
        :param i: flat index
        :return: 2D tuple index
        """
        return int(i / arr.shape[1]), i % arr.shape[1]

if __name__ == '__main__':
    size = 5

    _unmasked_pegs = np.tri(size, dtype=int)
    pegs = ma.masked_array(_unmasked_pegs, mask=np.tri(size, dtype=bool, k=-1).T, hard_mask=True)
    indices = np.tril_indices_from(pegs)
    indices = list(zip(*indices))

    flat_indices = [index_flat(pegs, i) for i in indices]
    pegs[0, 0] = 0
    pegs[2, 1] = 0

    print(pegs)
    print(np.rot90(np.flipud(pegs)))
    x = np.rot90(_unmasked_pegs)
    for i in range(len(x)-1, -1, -1):
        x[i] = np.roll(x[i],  -(len(x)-1 - i))
    pegs = ma.masked_array(x, mask=np.tri(size, dtype=bool, k=-1).T, hard_mask=True)
    print(pegs)

    if pegs[1, 3]:
        print("a")

    if not pegs[1, 3]:
        print("b")

    if pegs[1, 3] == 0:
        print("c")

    pass
