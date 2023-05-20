import numpy as np


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def trainTestCNN(data):
    indexes = [i for i in range(len(data))]
    trainSample = np.random.choice(indexes, int(0.8 * len(data)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]
    train = [data[i] for i in trainSample]
    test = [data[i] for i in testSample]
    return train, test
