from keras.layers import Dropout
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.layers import MaxPool2D, Flatten, Dense, Conv2D
from tensorflow.python.keras import Sequential

import numpy as np

from lab9.utils.load_data import loadDigitData, loadFlowersData
from lab9.neural_network.evaluations import predictByTool, predictByMe, evaluate
from lab9.neural_network.normalisations import normalisation, normalisationCNN
from lab9.utils.split_data import splitData, splitDataCNN
from sklearn import neural_network

from lab9.sepia_ANN.load_pics import loadPics, loadPictures
from lab9.utils.flatteners import flattenData
from lab9.utils.plot_cm import plot_confusion_matrix


def digits():
    inputs, outputs, outputNames = loadDigitData()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    trainInputs, testInputs = flattenData(trainInputs, testInputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    computedOutputs = predictByTool(trainInputs, trainOutputs, testInputs, testOutputs)
    print('Computed: ', list(computedOutputs))
    print('Real: ', testOutputs)
    computedOutputsByMe = predictByMe(trainInputs, trainOutputs, testInputs)
    print('Computed by me: ', computedOutputsByMe)
    print('Real: ', testOutputs)
    accuracy, precision, recall, confusion_matrix_by_me = evaluate(np.array(testOutputs), np.array(computedOutputsByMe), outputNames, division=36)
    plot_confusion_matrix(confusion_matrix_by_me, outputNames, "Digits - makeshift ANN")


def iris():
    inputs, outputs, outputNames, feature1, feature2, feature3, feature4, featureNames = loadFlowersData()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    computedOutputs = predictByTool(trainInputs, trainOutputs, testInputs, testOutputs)
    print('Computed: ', list(computedOutputs))
    print('Real: ', testOutputs)
    computedOutputsByMe = predictByMe(trainInputs, trainOutputs, testInputs)
    print('Computed by me: ', computedOutputsByMe)
    print('Real: ', testOutputs)
    accuracy, precision, recall, confusion_matrix_by_me = evaluate(np.array(testOutputs), np.array(computedOutputsByMe), outputNames, division=10)
    plot_confusion_matrix(confusion_matrix_by_me, outputNames, "Iris - makeshift ANN")


def filters():
    inputs, outputs, labels = loadPics()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)

    trainInputs = np.array(trainInputs)
    trainOutputs = np.array(trainOutputs)
    testInputs = np.array(testInputs)
    testOutputs = np.array(testOutputs)

    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(12, 25, 12), max_iter=5000)

    classifier.fit(trainInputs, trainOutputs)
    predictedLabels = classifier.predict(testInputs)
    acc, prec, recall, cm = evaluate(np.array(testOutputs), predictedLabels, labels, division=10)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    plot_confusion_matrix(cm, labels, 'Sepia ANN sklearn')


def filtersCNN():
    data = loadPictures('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul '
                        '4\\AI\\lab2\\lab9\\sepia_ANN\\data', 64)
    train, test, = splitDataCNN(data)
    trainInputs, trainOutputs, testInputs, testOutputs = normalisationCNN(train, test)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(trainInputs, trainOutputs, epochs=50, validation_data=(testInputs, testOutputs))
    computed = model.predict(x=testInputs)
    computed = [list(elem).index(max(list(elem))) for elem in computed]
    acc, prec, recall, cm = evaluate(testOutputs, computed, ['!Sepia', 'Sepia'], division=20)
    plot_confusion_matrix(cm, ['!Sepia', 'Sepia'], 'Sepia CNN keras')


if __name__ == '__main__':
    digits()
    iris()
    filters()
    filtersCNN()
    print('Hello World!')
