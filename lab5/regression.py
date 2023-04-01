from numpy import sqrt

from lab5.utils import plotRegression


def regression(outputs):
    real = []
    predicted = []
    for item in outputs:
        for i in range(3):
            real += [int(item[i])]
        for j in range(3, len(item)):
            predicted += [int(item[j])]

    subList = [real[n:n + 3] for n in range(0, len(real), 3)]
    subListP = [predicted[n:n + 3] for n in range(0, len(predicted), 3)]

    plotRegression(subList, subListP)

    mae = 0
    rmse = 0
    for real, predicted in zip(subList, subListP):
        mae += sum(abs(r - p) for r, p in zip(real, predicted)) / len(real)
        rmse += sqrt(sum((r - p) ** 2 for r, p in zip(real, predicted)) / len(real))

    return mae / len(subList), rmse / len(subList)
