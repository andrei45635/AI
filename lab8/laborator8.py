import numpy as np
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from lab8.myLogisticRegression import MyLogisticRegression
from lab8.normalisation import normalisation
from lab8.utils.plotters import plotInputs, plotPredictionsSepals, plotPredictionsPetals


if __name__ == '__main__':
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

    plotInputs(outputNames, outputs, inputs, feature1, feature2, feature3, feature4)

    # split data into train and test subsets
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # normalise the features
    trainInputs, testInputs = normalisation(trainInputs, testInputs)

    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]
    feature3train = [ex[2] for ex in trainInputs]
    feature4train = [ex[3] for ex in trainInputs]
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]
    feature3test = [ex[2] for ex in testInputs]
    feature4test = [ex[3] for ex in testInputs]

    plotInputs(outputNames, trainOutputs, trainInputs, feature1train, feature2train, feature3train, feature4train, 'normalised data')
    print('Classification models: (using tool)\n')
    labels = [label for label in set(outputs)]
    classifier = linear_model.LogisticRegression()
    classifier.fit(trainInputs, trainOutputs)

    w0, w1, w2, w3, w4 = classifier.intercept_, [classifier.coef_[_][0] for _ in range(len(labels))], [
        classifier.coef_[_][1] for _ in
        range(len(labels))], [classifier.coef_[_][2]
                              for _ in
                              range(len(labels))], [
        classifier.coef_[_][3] for _ in range(len(labels))]
    for _ in range(len(labels)):
        print(f"y{str(_ + 1)}(feat1, feat2, feat3, feat4) = {w0[_]} + {w1[_]} * feat1 + {w2[_]} * feat2 + {w3[_]} * feat3 + {w4[_]} * feat4")

    computedTestOutputsTool = classifier.predict(testInputs)

    error = 1 - accuracy_score(testOutputs, computedTestOutputsTool)
    print("classification error (tool): ", error)

    error = 0.0
    for t1, t2 in zip(computedTestOutputsTool, testOutputs):
        if t1 != t2:
            error += 1
    error = error / len(testOutputs)
    print("classification error (manual): ", error)

    print('accuracy score (sklearn): ', accuracy_score(testOutputs, computedTestOutputsTool))

    plotPredictionsSepals(feature1test, feature2test, testOutputs, computedTestOutputsTool, 'predicted sepals tool', outputNames)
    plotPredictionsPetals(feature3test, feature4test, testOutputs, computedTestOutputsTool, 'predicted petals tool', outputNames)

    print('Classification models: (using my code)\n')
    labels = [label for label in set(outputs)]
    classifier = MyLogisticRegression()
    classifier.fit(trainInputs, trainOutputs)
    w0, w1, w2, w3, w4 = classifier.intercept_, [classifier.coef_[2][0] for _ in range(len(labels))], [
        classifier.coef_[2][1] for _ in
        range(len(labels))], [classifier.coef_[2][2]
                              for _ in
                              range(len(labels))], [
        classifier.coef_[2][3] for _ in range(len(labels))]

    for _ in range(len(labels)):
        print(f"y{str(_ + 1)}(feat1, feat2, feat3, feat4) = {w0[_]} + {w1[_]} * feat1 + {w2[_]} * feat2 + {w3[_]} * feat3 + {w4[_]} * feat4")

    computedTestOutputs = classifier.predict(testInputs)

    error = 1 - accuracy_score(testOutputs, computedTestOutputs)
    print("classification error (tool): ", error)

    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        if t1 != t2:
            error += 1
    error = error / len(testOutputs)
    print("classification error (manual): ", error)

    print('accuracy score (my code): ', accuracy_score(testOutputs, computedTestOutputs))

    plotPredictionsSepals(feature1test, feature2test, testOutputs, computedTestOutputs, 'predicted sepals my own code', outputNames)
    plotPredictionsPetals(feature3test, feature4test, testOutputs, computedTestOutputs, 'predicted petals my own code', outputNames)
