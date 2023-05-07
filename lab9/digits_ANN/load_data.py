from sklearn.datasets import load_digits
import numpy as np


def loadDigitData():
    digits = load_digits()
    inputs = digits.images
    outputs = digits['target']
    outputNames = digits['target_names']

    noOfData = len(inputs)
    perm = np.random.permutation(noOfData)
    inputs = inputs[perm]
    outputs = outputs[perm]
    return inputs, outputs, outputNames
