def initMatrix(rows, cols):
    matrix = []
    for _ in range(rows):
        matrix += [[0 for _ in range(cols)]]
    return matrix


def multiplyMatrices(matrix1, matrix2):
    res = initMatrix(len(matrix1), len(matrix2[0]))
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                res[i][j] += matrix1[i][k] * matrix2[k][j]
    return res


def getIdentityMatrix(rows, cols):
    identity = initMatrix(rows, cols)
    for i in range(rows):
        identity[i][i] = 1
    return identity


def copyMatrix(matrix):
    mc = initMatrix(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            mc[i][j] = matrix[i][j]
    return mc


def invertMatrix(matrix):
    mc = copyMatrix(matrix)
    im = getIdentityMatrix(len(matrix), len(matrix[0]))
    imc = copyMatrix(im)
    indices = list(range(len(matrix)))
    for fd in range(len(matrix)):
        fdScaler = 1 / mc[fd][fd]
        for j in range(len(matrix)):
            mc[fd][j] *= fdScaler
            imc[fd][j] *= fdScaler
        for i in indices[0:fd] + indices[fd + 1:]:
            crScaler = mc[i][fd]
            for j in range(len(matrix)):
                mc[i][j] = mc[i][j] - crScaler * mc[fd][j]
                imc[i][j] = imc[i][j] - crScaler * imc[fd][j]
    return imc


def my_regression(gdp, freedom, happiness):
    x = [[1, el1, el2] for el1, el2 in zip(gdp, freedom)]
    x_transpose = list(map(list, zip(*x)))
    first_mp = multiplyMatrices(x_transpose, x)
    inv_mx = invertMatrix(first_mp)
    second_mp = multiplyMatrices(inv_mx, x_transpose)
    happiness_matrix = [[el] for el in happiness]
    res = multiplyMatrices(second_mp, happiness_matrix)
    w0, w1, w2 = res[0][0], res[1][0], res[2][0]
    modelstr = 'the learnt model: f(x) = ' + str(w0) + ' + ' + str(w1) + ' * x1' + ' + ' + str(w2) + ' * x2'
    return modelstr
