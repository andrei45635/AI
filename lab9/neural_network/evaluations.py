import numpy as np
from sklearn import neural_network
from sklearn.metrics import confusion_matrix, accuracy_score

from lab9.neural_network.my_neural_network import MyNN


def predictByTool(trainInputs, trainOutputs, testInputs, testOutputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100, solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.1)
    classifier.fit(trainInputs, trainOutputs)
    computedOutputs = classifier.predict(testInputs)
    print('Accuracy using sk-learn: ', classifier.score(testInputs, testOutputs))
    return computedOutputs


def predictByMe(trainInputs, trainOutputs, testInputs):
    classifier = MyNN(hiddenLayers=10)
    classifier.fit(np.array(trainInputs), np.array(trainOutputs))
    computedOutputs = classifier.predict(testInputs)
    return computedOutputs


def evaluate(testOutputs, predictedLabels, labels, division):
    cmx = confusion_matrix(testOutputs, predictedLabels)
    accuracy = accuracy_score(testOutputs, predictedLabels)
    # accuracy = sum([cmx[i][i] for i in range(len(predictedLabels) // division)]) / len(testOutputs)
    precision = {}
    recall = {}
    for i in range(len(labels)):
        precision[labels[i]] = cmx[i][i] / sum([cmx[j][i] for j in range(len(labels))])
        recall[labels[i]] = cmx[i][i] / sum([cmx[i][j] for j in range(len(labels))])
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    return accuracy, precision, recall, cmx
