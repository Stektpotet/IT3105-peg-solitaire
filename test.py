import matplotlib.pyplot as plt
import argparse as arg
import numpy as np
import numpy.ma as ma
# A runnable file to test out various things
if __name__ == '__main__':

    a = np.tri(5, dtype=int)
    x = ma.masked_array(a, mask=np.tri(5, dtype=bool, k=-1).T, hard_mask=True)

    x[int(x.shape[0] / 2), int(x.shape[1] / 2) - int(x.shape[1] / 4)] = 0
    x[0, 0] = 0
    x[1, 0] = 0

    print(x)

    p = np.where(x == 0)
    for k in zip(*p):
        print(k)

    pass
