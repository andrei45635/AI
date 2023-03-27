import random
from random import randint


def generatePermutation(n):
    perm = [i for i in range(1, n+1)]
    pos1 = randint(0, len(perm) - 1)
    pos2 = randint(0, len(perm) - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm


def generatePermsFixed(n, start, end):
    perm = [i for i in range(1, n + 1)]
    perm.remove(start)
    perm.remove(end)
    random.shuffle(perm)
    perm.insert(0, start)
    perm.append(end)
    # perm.insert(n, end)
    return perm
