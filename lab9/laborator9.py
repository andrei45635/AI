import numpy as np
from matplotlib import pyplot as plt

from lab9.digits_ANN.load_data import loadDigitData
from lab9.digits_ANN.normalisation import normalisation, normalize
from lab9.digits_ANN.split_data import splitData
from sklearn import neural_network
from sklearn.metrics import confusion_matrix

from lab9.iris_ANN.iris import loadIris
from lab9.sepia_ANN.load_pics import loadPics


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


def flatten1(mat):
    x = []
    for line in mat:
        for el in line:
            for k in el:
                x.append(k)
    return x


def evalMultiClass(realLabels, computedLabels, labels):
    print(labels)
    print(realLabels)
    confusion = confusion_matrix(realLabels, computedLabels)
    accr = sum([confusion[i][i] for i in range(len(labels))]) / len(realLabels)
    precision = {}
    rcl = {}
    for i in range(len(labels)):
        print('cc', confusion[i][i])
        precision[labels[i]] = confusion[i][i] / sum([confusion[j][i] for j in range(len(labels))])
        rcl[labels[i]] = confusion[i][i] / sum([confusion[i][j] for j in range(len(labels))])
    return accr, precision, rcl, confusion


def digits():
    inputs, outputs, outputNames = loadDigitData()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    trainInputsFlattened = [flatten(el) for el in trainInputs]
    testInputsFlattened = [flatten(el) for el in testInputs]

    normalisedTrainData, normalisedTestData = normalisation(trainInputsFlattened, testInputsFlattened)

    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100,
                                              solver='sgd',
                                              verbose=10, random_state=1, learning_rate_init=.1)

    classifier.fit(normalisedTrainData, trainOutputs)
    predictedLabels = classifier.predict(normalisedTestData)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)

    # plot first 50 test images and their real and computed labels
    n = 10
    m = 5
    fig, axes = plt.subplots(n, m, figsize=(7, 7))
    fig.tight_layout()
    for i in range(0, n):
        for j in range(0, m):
            axes[i][j].imshow(testInputs[m * i + j])
            if testOutputs[m * i + j] == predictedLabels[m * i + j]:
                font = 'normal'
            else:
                font = 'bold'
            axes[i][j].set_title(
                'real ' + str(testOutputs[m * i + j]) + '\npredicted ' + str(predictedLabels[m * i + j]),
                fontweight=font)
            axes[i][j].set_axis_off()

    plt.show()


def filters():
    inputs, outputs, labels = loadPics()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    print(len(trainInputs), len(trainOutputs))
    trainInputsFlattened = [flatten1(el) for el in trainInputs]
    testInputsFlattened = [flatten1(el) for el in testInputs]

    normalisedTrainData = normalize(trainInputsFlattened)
    normalisedTestData = normalize(testInputsFlattened)

    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=200,
                                              solver='sgd',
                                              verbose=10, random_state=1, learning_rate_init=.1)

    classifier.fit(normalisedTrainData, trainOutputs)
    predictedLabels = classifier.predict(normalisedTestData)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, labels)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)


def iris():
    nn = loadIris()



if __name__ == '__main__':
    # digits()
    # filters()
    iris()
