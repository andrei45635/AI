import csv

import numpy as np
from matplotlib import pyplot as plt


def readFromCSV(fileName):
    with open(fileName, 'r') as f:
        csvreader = csv.reader(f)
        dt = []
        for row in csvreader:
            dt += [row]
        return dt


def readTxt(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        real = []
        predicted = []
        for line in lines:
            real += [int(line.strip()[0])]
            predicted += [int(line.strip()[2])]
        return real, predicted


def readFile(fileName):
    with open(fileName) as f:
        real = []
        predicted = []
        for line in f.readlines():
            real += [float(line.strip().split()[0])]
            predicted += [float(line.strip().split()[1])]
        return real, predicted


def readMultiClass(fileName):
    with open(fileName) as f:
        realProbs = []
        for line in f.readlines():
            probs = list(map(float, line.strip().split()))
            realProbs += probs
        subList1 = [realProbs[n:n + 3] for n in range(0, len(realProbs), 3)]
        return np.array(subList1, dtype=float)


def readMultiTarget(fileName):
    with open(fileName) as f:
        realProbs = []
        for line in f.readlines():
            probs = list(map(float, line.strip().split()))
            realProbs += probs
        subList1 = [realProbs[n:n + 5] for n in range(0, len(realProbs), 5)]
        return np.array(subList1, dtype=float)


def getSubLists(outputs):
    real = []
    predicted = []
    for item in outputs:
        for i in range(3):
            real += [int(item[i])]
        for j in range(3, len(item)):
            predicted += [int(item[j])]

    subList = [real[n:n + 3] for n in range(0, len(real), 3)]
    subListP = [predicted[n:n + 3] for n in range(0, len(predicted), 3)]
    return subList, subListP


def getRealValues(subList):
    realWeights = [w[0] for w in subList]
    realWaist = [w[1] for w in subList]
    realPulse = [w[2] for w in subList]

    return realWeights, realWaist, realPulse


def getPredictedValues(subListP):
    predictedWeights = [pw[0] for pw in subListP]
    predictedWaist = [pw[1] for pw in subListP]
    predictedPulse = [pw[2] for pw in subListP]

    return predictedWeights, predictedWaist, predictedPulse


def plotRegression(subList, subListP):
    realWeights, realWaist, realPulse = getRealValues(subList)
    predictedWeights, predictedWaist, predictedPulse = getPredictedValues(subListP)

    indexesW = [i for i in range(len(realWeights))]
    indexesWs = [i for i in range(len(realWaist))]
    indexesP = [i for i in range(len(realPulse))]

    realW, = plt.plot(indexesW, realWeights, 'ro', label='real weights')
    realWs, = plt.plot(indexesWs, realWaist, 'g+', label='real waists')
    realP, = plt.plot(indexesP, realPulse, 'y*', label='real pulses')

    predictedW, = plt.plot(indexesW, predictedWeights, 'bo', label='predicted weights')
    predictedWs, = plt.plot(indexesWs, predictedWaist, 'y+', label='predicted waists')
    predictedP, = plt.plot(indexesP, predictedPulse, 'g*', label='predicted pulses')

    plt.legend([realW, realWs, realP, (realW, realWs, realP, predictedW, predictedWs, predictedP)],
               ["Real", "Predicted"])
    plt.show()
