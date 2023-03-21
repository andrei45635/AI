from numpy.random import randint
import numpy as np


def generateBinaryVector(n):
    vect = [0 for _ in range(n)]
    p = 0.2
    pos1 = randint(0, len(vect) - 1)
    pos2 = randint(0, len(vect) - 1)
    if np.rand() > p:
        vect[pos1], vect[pos2] = 1, 1
    return vect
