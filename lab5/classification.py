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

    for label in labelNames:
        TP = sum([1 if real[i] == label and predicted[i] == label else 0 for i in range(len(real))]) / len(real)
        TN = sum([1 if real[i] != label and predicted[i] != label else 0 for i in range(len(real))]) / len(real)
        FP = sum([1 if real[i] != label and predicted[i] == label else 0 for i in range(len(real))]) / len(real)
        FN = sum([1 if real[i] == label and predicted[i] != label else 0 for i in range(len(real))]) / len(real)

    precisionPos = TP / (TP + FP)
    recallPos = TP / (TP + FN)
    precisionNeg = TN / (TN + FN)
    recallNeg = TN / (TN + FP)
    return acc, precisionPos, precisionNeg, recallPos, recallNeg

