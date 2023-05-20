import math
from queue import Queue
import heapq


# Determina ultimul cuvant dpdv alfabetic dintr-un string dat 
# txt : string = textul dat
# return max = ultimul cuvant dpdv lexicografic (string)
def last_word(txt: str) -> str:
    words = txt.split()
    max = ""
    for i in range(len(words) - 1):
        if words[i] > max:
            max = words[i]
    if words[len(words) - 1] > max:
        max = words[len(words) - 1]
    return max


print("Problema 1")
print(last_word("Ana are mere rosii si galbene"))
print(last_word("a b c d e f g h i j k l m n o p q r s t u x y w z"))


# Determina distanta euclidiana dintre 2 puncte
# p: list[int] = primul punct
# q: list[int] = al doilea punct
# return distance : float = distanta euclidiana dintre p si q
def euclidean_distance(p, q) -> float:
    return math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)


print("Problema 2")
print(euclidean_distance([1, 5], [4, 1]))


# Determina produsul scalar a doi vectori rari
# vect1: list[int] = primul vector
# vect2: list[int] = al doilea vector
# return dot_product: float = produsul scalar
def dot_product(vect1, vect2) -> float:
    dot_product = 0
    for i in range(len(vect1)):
        dot_product += vect1[i] * vect2[i]
    return dot_product


print("Problema 3")
print(dot_product([1, 0, 2, 0, 3], [1, 2, 0, 3, 1]))


# Determina cuvintele care apar exact o singura data
# txt: string = textul dat
# return least: list[str] = cuvintele care apar o singura data
def least_words(txt):
    words_freq = {}
    least = []
    for word in txt.split():
        words_freq[word] = txt.split().count(word)
    for word, freq in words_freq.items():
        if freq == 1:
            least.append(word)
    return least


print("Problema 4")
print(least_words("ana are ana are mere rosii ana"))
print(least_words("ana"))


# Determina valoarea care se repeta de exact 2 ori
# vect: list[int] = vectorul dat (i e vect | i e {1, 2, ..., n - 1})
# return val: int = valoarea care se repeta de exact 2 ori
def two_times(vect) -> int:
    freq = {}
    for i in vect:
        freq[i] = vect.count(i)
    for k, v in freq.items():
        if v == 2:
            return k
    return 0


print("Problema 5")
print(two_times([1, 2, 3, 4, 2]))


# Determina elementul majoritar dintr-un vector dat
# vect: list[int] = vectorul dat 
# return val: int = elementul care a aparut de > len(vect)/2 ori
def majoritar(vect) -> int:
    freq = {}
    for i in vect:
        freq[i] = vect.count(i)
    for k, v in freq.items():
        if v > len(vect) // 2:
            return k
    return 0


print("Problema 6")
print(majoritar([2, 8, 7, 2, 2, 5, 2, 3, 2, 1, 2, 2]))


# Determina al k-lea cel mai mare element dintr-un vector
# vect: list[int] = vectorul dat
# return val: int = al k-lea cel mai mare element
def kth_elem(vect, k) -> int:
    return sorted(vect, reverse=True)[k - 1]


print("Problema 7_1")
print(kth_elem([7, 4, 6, 3, 9], 2))


def kth_elem2(vect, k) -> int:
    hp = vect[0:k]
    heapq.heapify(hp)
    for i in range(k, len(vect)):
        if vect[i] > hp[0]:
            heapq.heapreplace(hp, vect[i])
    return hp[0]


print("Problema 7_2")
print(kth_elem2([7, 4, 6, 3, 9], 2))
print(kth_elem2([5, 4, 6, 3, 9], 2))


# Genereaza numerele in reprezentare binara de la 1 la n dat
# n: int = limita superioara a intervalului 1...n 
# return val: str = valoare binara
def binary_vals(n: int):
    q = Queue()
    q.put("1")
    while (n > 0):
        n -= 1
        b1 = q.get()
        print(b1)
        b2 = b1
        q.put(b1 + "0")
        q.put(b2 + "1")


print("Problema 8")
print(binary_vals(4))


# Determina suma elementelor din submatricea data de coordonatele punctelor p si q
# p: list[int] = primul punct
# q: list[int] = al doilea punct 
# return suma: int = suma ceruta
def submatrix_sum(matrix, p, q) -> sum:
    s = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if p[0] <= i <= q[0] and p[1] <= j <= q[1]:
                s += matrix[i][j]
    return s


print("Problema 9")
print(submatrix_sum([[0, 2, 5, 4, 1], [4, 8, 2, 3, 7], [6, 3, 4, 6, 2], [7, 3, 1, 8, 3], [1, 5, 7, 9, 4]], [1, 1],
                    [3, 3]))
print(submatrix_sum([[0, 2, 5, 4, 1], [4, 8, 2, 3, 7], [6, 3, 4, 6, 2], [7, 3, 1, 8, 3], [1, 5, 7, 9, 4]], [2, 2],
                    [4, 4]))


# Determina indexul liniei care contine cele mai multe elemente de 1
# matrix: list[list[int]] = matricea data (formata doar din 0 si 1)
# return index: int = indexul liniei cu cei mai multi de 1 
def most_ones(matrix) -> int:
    sums = [sum(e) for e in matrix]
    for i in range(len(matrix)):
        s = 0
        for j in range(len(matrix[i])):
            s += matrix[i][j]
        if s == max(sums):
            return i
    return 0


print("Problema 10")
print(most_ones([[0, 0, 0, 1, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]))

# Genereaza matricea in care au fost inlocuite cu 1 toate insulele de 0
# matrix: list[list[int]] = matricea data
# return replaced: list[list[int]] = matricea rezultata
matrix = [[1, 1, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
          [1, 1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]]


def dfs(i: int, j: int) -> None:
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or matrix[i][j] != 0:
        return

    matrix[i][j] = -1

    dfs(i + 1, j)
    dfs(i - 1, j)
    dfs(i, j + 1)
    dfs(i, j - 1)


def ones(matrix):
    for i in range(len(matrix)):
        dfs(i, 0)
        dfs(i, len(matrix[0]) - 1)

    for j in range(len(matrix[0])):
        dfs(0, j)
        dfs(len(matrix) - 1, j)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                matrix[i][j] = 1
            elif matrix[i][j] == -1:
                matrix[i][j] = 0
    return matrix


print("Problema 11")
print(ones(matrix))
