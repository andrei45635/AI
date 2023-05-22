import numpy as np
from sklearn.preprocessing import StandardScaler


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def normalisationCNN(trainData, testData):
    trainInputs, trainOutputs, testInputs, testOutputs = [], [], [], []
    for feat, label in trainData:
        trainInputs.append(feat)
        trainOutputs.append(label)
    for feat, label in testData:
        testInputs.append(feat)
        testOutputs.append(label)

    trainInputs = np.array(trainInputs) / 255.0
    testInputs = np.array(testInputs) / 255.0
    trainInputs.reshape(-1, 64, 64, 1)
    trainOutputs = np.array(trainOutputs)
    testOutputs = np.array(testOutputs)
    testInputs.reshape(-1, 64, 64, 1)
    return trainInputs, trainOutputs, testInputs, testOutputs