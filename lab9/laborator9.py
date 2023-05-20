from keras.layers import Dropout
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.layers import MaxPool2D, Flatten, Dense, Conv2D
from tensorflow.python.keras import Sequential

import numpy as np
from matplotlib import pyplot as plt

from lab9.digits_ANN.load_data import loadDigitData
from lab9.sepia_ANN.split_data import trainTestCNN
from lab9.utils.normalisation import normalisation, normalisationCNN
from lab9.digits_ANN.split_data import splitData
from sklearn import neural_network

from lab9.iris_ANN.iris import loadIris
from lab9.sepia_ANN.load_pics import loadPics, loadPictures
from lab9.utils.eval import evalMultiClass, flatten
from lab9.utils.plot_cm import plot_confusion_matrix


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
    trainIns = []
    for i in trainInputs:
        if i.shape != (388800,):
            continue
        trainIns.append(i.reshape((388800,)))

    trainInputs = np.array(trainIns)
    trainOutputs = np.array(trainOutputs)
    testInputs = np.array(testInputs)
    testOutputs = np.array(testOutputs)

    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(12, 25, 12), max_iter=10000)

    classifier.fit(trainInputs, trainOutputs)
    predictedLabels = classifier.predict(testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, labels)

    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    plot_confusion_matrix(cm, labels, 'Sepia ANN')


def filtersCNN():
    data = loadPictures('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul 4\\AI\\lab2\\lab9\\sepia_ANN\\images', 64)
    train, test = trainTestCNN(data)
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
    testOutputs = np.where(testOutputs >= 0.0, 1, 0)
    acc, prec, recall, cm = evalMultiClass(testOutputs, computed, ['!Sepia', 'Sepia'])
    plot_confusion_matrix(cm, ['!Sepia', 'Sepia'], 'Sepia CNN')


def iris():
    nn = loadIris()


if __name__ == '__main__':
    # digits()
    # filters()
    filtersCNN()
    # iris()
    print('Hello World!')
