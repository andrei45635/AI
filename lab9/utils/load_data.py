from sklearn.datasets import load_digits, load_iris


def loadDigitData():
    digits = load_digits()
    inputs = digits.images
    outputs = digits['target']
    outputNames = digits['target_names']
    return inputs, outputs, outputNames


def loadFlowersData():
    data = load_iris()
    input = data['data']
    output = data['target']
    outputNames = data['target_names']
    features = data['feature_names']
    feature1 = [feat[features.index('sepal length (cm)')] for feat in input]
    feature2 = [feat[features.index('sepal width (cm)')] for feat in input]
    feature3 = [feat[features.index('petal length (cm)')] for feat in input]
    feature4 = [feat[features.index('petal width (cm)')] for feat in input]
    input = [[feat[features.index('sepal length (cm)')],
              feat[features.index('sepal width (cm)')],
              feat[features.index('petal length (cm)')],
              feat[features.index('petal width (cm)')]] for feat in input]
    return input, output, outputNames, feature1, feature2, feature3, feature4, features
