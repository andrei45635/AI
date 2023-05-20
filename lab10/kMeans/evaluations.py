import numpy as np
from sklearn import neural_network
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from lab10.kMeans.my_kMeans import AndreiKMeans


def predictByTool(trainFeatures, testFeatures, labels, classes):
    unsupervisedClassifier = KMeans(n_clusters=classes, random_state=0)
    unsupervisedClassifier.fit(trainFeatures)
    computedIndexes = unsupervisedClassifier.predict(testFeatures)
    computedOutputs = [labels[val] for val in computedIndexes]
    return computedOutputs


def predictByMe(trainFeatures, testFeatures, labels, classes):
    myUnsupervisedClassifier = AndreiKMeans(nClusters=classes)
    myUnsupervisedClassifier.fit(trainFeatures)
    centroids, computedIndexes = myUnsupervisedClassifier.evaluate(testFeatures)
    computedOutputs = [labels[val] for val in computedIndexes]
    return computedOutputs, centroids, computedIndexes


def predictSupervised(trainInputs, trainOutputs, testInputs):
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(25, 40, 20), activation='relu', max_iter=1000,
                                              solver='sgd',
                                              verbose=0, random_state=1, learning_rate_init=.01)
    classifier.fit(trainInputs, trainOutputs)
    computedOutputs = classifier.predict(testInputs)
    return computedOutputs


def predictHybrid(trainInputs, trainOutputs, testInputs, testOutputs):  # semi-supervised
    n = 100  # only 100 inputs will be labeled
    classifier = neural_network.MLPClassifier()
    classifier.fit(trainInputs[:n], trainOutputs[:n])
    computedOutputs = classifier.predict(testInputs)
    prevAcc = accuracy_score(testOutputs, computedOutputs)

    unsupervisedClassifier = KMeans(n_clusters=n, random_state=0)
    dists = unsupervisedClassifier.fit_transform(trainInputs)  # distance matrix points - centroids
    repIdxs = np.argmin(dists, axis=0)
    repInputs = [trainInputs[i] for i in repIdxs]
    repOutputs = [list(trainOutputs)[i] for i in repIdxs]
    classifier = neural_network.MLPClassifier()
    classifier.fit(repInputs, repOutputs)  # fit with the most representative data
    computedOutputs = classifier.predict(testInputs)
    return computedOutputs, prevAcc
