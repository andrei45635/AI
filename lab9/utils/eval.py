from sklearn.metrics import confusion_matrix


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
    confusion = confusion_matrix(realLabels, computedLabels)
    print(confusion)
    accr = sum([confusion[i][i] for i in range(len(labels))]) / len(realLabels)
    precision = {}
    rcl = {}
    for i in range(len(labels)):
        precision[labels[i]] = confusion[i][i] / sum([confusion[j][i] for j in range(len(labels))])
        rcl[labels[i]] = confusion[i][i] / sum([confusion[i][j] for j in range(len(labels))])
    return accr, precision, rcl, confusion