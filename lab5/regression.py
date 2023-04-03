from numpy import sqrt

from lab5.utils import plotRegression, getSubLists, getRealValues, getPredictedValues


def regression(outputs):
    subList, subListP = getSubLists(outputs)
    plotRegression(subList, subListP)

    mae = 0
    rmse = 0
    for real, predicted in zip(subList, subListP):
        mae += sum(abs(r - p) for r, p in zip(real, predicted)) / len(real)
        rmse += sqrt(sum((r - p) ** 2 for r, p in zip(real, predicted)) / len(real))

    return mae / len(subList), rmse / len(subList)


def regressionV2(outputs):
    subList, subListP = getSubLists(outputs)
    realWeights, realWaist, realPulse = getRealValues(subList)
    predictedWeights, predictedWaist, predictedPulse = getPredictedValues(subListP)

    errorL1_weights = sum(abs(r - p) for r, p in zip(realWeights, predictedWeights)) / len(realWeights)
    errorL1_waists = sum(abs(r - p) for r, p in zip(realWaist, predictedWaist)) / len(realWaist)
    errorL1_pulses = sum(abs(r - p) for r, p in zip(realPulse, predictedPulse)) / len(realPulse)

    errorL2_weights = sqrt(sum((r - p) ** 2 for r, p in zip(realWeights, predictedWeights)) / len(realWeights))
    errorL2_waists = sqrt(sum((r - p) ** 2 for r, p in zip(realWaist, predictedWaist)) / len(realWaist))
    errorL2_pulses = sqrt(sum((r - p) ** 2 for r, p in zip(realPulse, predictedPulse)) / len(realPulse))

    return errorL1_weights, errorL1_waists, errorL1_pulses, errorL2_weights, errorL2_waists, errorL2_pulses
