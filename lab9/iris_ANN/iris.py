import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from lab8.normalisation import normalisation


def loadIris():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featureNames = list(data['feature_names'])

    feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[featureNames.index('sepal width (cm)')] for feat in inputs]
    feature3 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    feature4 = [feat[featureNames.index('petal width (cm)')] for feat in inputs]
    inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('sepal width (cm)')],
               feat[featureNames.index('petal length (cm)')], feat[featureNames.index('petal width (cm)')]] for feat in
              inputs]
    inputs = np.array(inputs)

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # normalise the features
    # trainInputs, testInputs = normalisation(trainInputs, testInputs)

    f = len(trainInputs[0])  # no. of features
    o = len(trainOutputs)  # no. of classes

    layers = [f, 5, 10, o]  # no. of nodes
    L, E = 0.15, 100

    aa = neuralNetwork(trainInputs, trainOutputs, testInputs, testOutputs, epochs=E, nodes=layers, learningRate=L)
    return aa

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return np.multiply(x, 1 - x)


def initWeight(nodes):
    # randomly initialize the weights of the nodes in [-1, 1] (interval)
    layers, weights = len(nodes), []
    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for _ in range(nodes[i - 1] + 1)] for _ in range(nodes[i])]
        weights.append(np.matrix(w))
    return weights


def forwardPropagation(x, weights, layers):
    activations, layerInput = [x], x
    for i in range(layers):
        activation = sigmoid(np.dot(layerInput, np.transpose(weights[i])))
        activations.append(activation)
        layerInput = np.append(1, activation)
    return activations


def backwardPropagation(y, activations, layers, weights, learningRate):
    finalOutput = activations[-1]
    err = np.matrix(y - finalOutput)  # error after 1 cycle

    for i in range(layers, 0, -1):
        currActivation = activations[i]
        if i > 1:
            # append previous
            prevActivation = np.append(1, activations[i - 1])
        else:
            # first hidden layer
            prevActivation = activations[0]
        delta = np.multiply(err, sigmoidDerivative(currActivation))
        weights[i - 1] += learningRate * np.multiply(delta.T, prevActivation)
        wc = np.delete(weights[i - 1], [0], axis=1)
        err = np.dot(delta, wc)  # current layer error

    return weights


def train(x, y, learningRate, weights):
    layers = len(weights)
    for i in range(len(x)):
        x, y = x, y[i]
        x = np.matrix(np.append(1, x))

        activations = forwardPropagation(x, weights, layers)
        weights = backwardPropagation(y, activations, layers, weights, learningRate)

    return weights


def predict(item, weights, finalOutput):
    layers = len(weights)
    item = np.append(1, item)

    # forward propagation
    activations = forwardPropagation(item, weights, layers)

    Foutput = activations[-1].A1
    index = FindMaxActivation(finalOutput)

    y = [0 for _ in range(len(Foutput))]
    y[index] = 1

    return y


def FindMaxActivation(output):
    m, index = output[0], 0
    for i in range(1, len(output)):
        if output[i] > m:
            m, index = output[i], i

    return index


def accuracy(X, Y, weights):
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = predict(x, weights, [])
        if y == guess:
            # right prediction
            correct += 1

    return correct / len(X)


def neuralNetwork(trainInputs, trainOutputs, testInputs, testOutputs, epochs=100, nodes=None, learningRate=.1):
    if nodes is None:
        nodes = []

    weights = initWeight(nodes)

    for epoch in range(1, epochs + 1):
        weights = train(trainInputs, trainOutputs, learningRate, weights)
        if epoch % 20 == 0:
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(accuracy(trainInputs, trainOutputs, weights)))
            if trainInputs.any():
                print("Validation Accuracy:{}".format(accuracy(testInputs, testOutputs, weights)))

    return weights
