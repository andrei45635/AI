from sklearn.metrics import confusion_matrix


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


def flattenData(train, test):
    train_data = [flatten(data) for data in train]
    test_data = [flatten(data) for data in test]
    return train_data, test_data


def flatten1(mat):
    x = []
    for line in mat:
        for el in line:
            for k in el:
                x.append(k)
    return x
