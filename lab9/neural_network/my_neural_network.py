import numpy as np
from numpy import mean


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


class MyNN:
    def __init__(self, hiddenLayers=12, maxIters=7000, learningRate=.001):
        self.__hiddenLayers = hiddenLayers
        self.__maxIters = maxIters
        self.__learningRate = learningRate
        self.__weights, self.__loss = [], []

    def fit(self, x, y):
        noOfFeatures = len(x[0])
        noOfOutputs = len(set(y))
        # one-hot encoding
        tempY = np.zeros((len(y), noOfOutputs))
        for i in range(len(y)):
            tempY[i, y[i]] = 1
        y = tempY

        # assign random values to the weights and biases
        weightIH = np.random.rand(noOfFeatures, self.__hiddenLayers)
        biasIH = np.random.randn(self.__hiddenLayers)
        weightHO = np.random.rand(self.__hiddenLayers, noOfOutputs)
        biasHO = np.random.randn(noOfOutputs)

        for _ in range(self.__maxIters):
            # forward propagation
            yIH = np.dot(x, weightIH) + biasIH
            hiddenInputY = sigmoid(yIH)
            yHO = np.dot(hiddenInputY, weightHO) + biasHO
            outputY = softmax(yHO)

            # backward propagation
            err = outputY - y
            errWeightHO = np.dot(np.transpose(hiddenInputY), err)
            errBiasHO = err

            # error propagated back to the hidden layer
            errBackProg = np.dot(err, np.transpose(weightHO))
            dah = sigmoidDerivative(yIH)  # derivative of the activation function with respect to hidden layer output
            dwh = x

            # updating weights and biases
            errWeightIH = np.dot(np.transpose(dwh), dah * errBackProg)
            self.__loss += mean(errWeightIH)
            errBiasIH = errBackProg * dah
            weightIH -= self.__learningRate * errWeightIH
            biasIH -= self.__learningRate * errBiasIH.sum(axis=0)
            weightHO -= self.__learningRate * errWeightHO
            biasHO -= self.__learningRate * errBiasHO.sum(axis=0)

        self.__weights = [weightIH, biasIH, weightHO, biasHO]

    def predict(self, x):
        weightIH, biasIH, weightHO, biasHO = self.__weights
        yIH = np.dot(x, weightIH) + biasIH
        hiddenInputY = sigmoid(yIH)
        yHO = np.dot(hiddenInputY, weightHO) + biasHO
        outputY = softmax(yHO)
        computedOutput = [list(output).index(max(output)) for output in outputY]
        return computedOutput
