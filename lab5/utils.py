import csv

from matplotlib import pyplot as plt


def readFromCSV(fileName):
    with open(fileName, 'r') as f:
        csvreader = csv.reader(f)
        dt = []
        for row in csvreader:
            dt += [row]
        return dt


def plotRegression(subList, subListP):
    realWeights = [w[0] for w in subList]
    realWaist = [w[1] for w in subList]
    realPulse = [w[2] for w in subList]

    predictedWeights = [pw[0] for pw in subListP]
    predictedWaist = [pw[1] for pw in subListP]
    predictedPulse = [pw[2] for pw in subListP]

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


'''
        errorL1_weights = sum(abs(r - p) for r, p in zip(realWeights, predictedWeights)) / len(realWeights)
        errorL1_waists = sum(abs(r - p) for r, p in zip(realWaist, predictedWaist)) / len(realWaist)
        errorL1_pulses = sum(abs(r - p) for r, p in zip(realPulse, predictedPulse)) / len(realPulse)
        print('errorL1_weights: ', errorL1_weights)
        print('errorL1_waists: ', errorL1_waists)
        print('errorL1_pulses: ', errorL1_pulses)

        errorL2_weights = sqrt(sum((r - p) ** 2for r, p in zip(realWeights, predictedWeights)) / len(realWeights))
        errorL2_waists = sqrt(sum((r - p) ** 2 for r, p in zip(realWaist, predictedWaist)) / len(realWaist))
        errorL2_pulses = sqrt(sum((r - p) ** 2 for r, p in zip(realPulse, predictedPulse)) / len(realPulse))
        print('errorL2_weights: ', errorL2_weights)
        print('errorL2_waists: ', errorL2_waists)
        print('errorL2_pulses: ', errorL2_pulses)
'''