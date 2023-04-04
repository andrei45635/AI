import numpy as np
from numpy import log
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def getRealPredictedLabels(data):
    real = []
    predicted = []
    for row in data:
        real += [row[0]]
        predicted += [row[1]]
    return real, predicted


def classificationV1(real, predicted, labelNames):
    cm = confusion_matrix(real, predicted, labels=labelNames)
    acc = accuracy_score(real, predicted)
    precision = precision_score(real, predicted, average=None, labels=labelNames)
    rc = recall_score(real, predicted, average=None, labels=labelNames)
    return cm, acc, precision, rc


def classificationV2(real, predicted, labelNames):
    acc = sum([1 if real[i] == predicted[i] else 0 for i in range(len(real))]) / len(real)
    TP, TN, FP, FN = 0, 0, 0, 0
    precisionPos, recallPos = 0, 0
    for label in labelNames:
        TP = sum([1 if real[i] == label and predicted[i] == label else 0 for i in range(len(real))]) / len(real)
        TN = sum([1 if real[i] != label and predicted[i] != label else 0 for i in range(len(real))]) / len(real)
        FP = sum([1 if real[i] != label and predicted[i] == label else 0 for i in range(len(real))]) / len(real)
        FN = sum([1 if real[i] == label and predicted[i] != label else 0 for i in range(len(real))]) / len(real)
        precisionPos = TP / (TP + FP)
        recallPos = TP / (TP + FN)
        print(label, precisionPos)
        print(label, recallPos)
    # precisionPos = TP / (TP + FP)
    # recallPos = TP / (TP + FN)
    precisionNeg = TN / (TN + FN)
    recallNeg = TN / (TN + FP)
    return acc, precisionPos, precisionNeg, recallPos, recallNeg


def lossClassification(real, predicted):
    """
    Formula for the binary cross-entropy loss function
        -> L = -[y * log(p + epsilon) + (1-y) * log(1-p + epsilon)]
    """
    entropy = 0
    epsilon = np.finfo(np.float32).eps
    for rl, pr in zip(real, predicted):
        entropy += -(int(rl) * log(int(pr) + epsilon) + (1 - int(rl)) * log(1 - int(pr) + epsilon))
    return entropy / len(real)


def lossV2(real, predicted, realProb, predictedProb):
    entropy = 0
    for rProb, pProb in zip(realProb, predictedProb):
        for rl, pd in zip(real, predicted):
            entropy += -(rl * log(pProb) + (1 - rl) * log(1 - pProb))
    entropy1 = 0
    for rl, pProb in zip(real, predictedProb):
        entropy1 += -(rl * log(pProb) + (1 - rl) * log(1 - pProb))
    # print(entropy1/len(real))
    return entropy / len(real)


def multiClassLoss(probArr, predArr):
    """
    The formula for the categorical cross-entropy loss is:
    L = -∑(y_true * log(y_pred))
    where:
        -> y_true is a one-hot encoded vector representing the true class label (i.e., a vector with a 1 in the position corresponding to the true class and 0s elsewhere)
        -> y_pred is a vector of predicted class probabilities (i.e., a vector of length equal to the number of classes, with each element representing the predicted probability of the corresponding class)
    """
    epsilon = np.finfo(np.float32).eps
    return -np.mean(np.sum(probArr * np.log(predArr + epsilon), axis=1))


def multiTargetLoss(probArr, predArr):
    """
    The formula for the binary cross-entropy loss is:
        -> L = -∑(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
    where:
        -> y_true is a binary vector representing the true labels (i.e., a vector with 1s in the positions corresponding to the true labels and 0s elsewhere)
        -> y_pred is a vector of predicted label probabilities (i.e., a vector of length equal to the number of labels, with each element representing the predicted probability of the corresponding label)
    """
    epsilon = np.finfo(np.float32).eps
    return -np.mean(np.sum(probArr * np.log(predArr + epsilon) + (1 - probArr) * np.log(1 - predArr + epsilon), axis=1))
