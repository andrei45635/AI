import glob
from PIL import Image

import numpy as np


def loadPics():
    inputs, outputs = [], []
    labels = ['!Sepia', 'Sepia']
    size = (360, 360)

    for file in glob.iglob('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul 4\\AI\\lab2\\lab9\\sepia_ANN\\images\\non_sepia\\*'):
        img = Image.open(file).resize(size)
        if img.mode != 'RGB':
            img = img.convert('RGB')  # convert grayscale images to RGB
        inputs.append(np.asarray(img))
        outputs.append(0)

    for file in glob.iglob('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul 4\\AI\\lab2\\lab9\\sepia_ANN\\images\\sepia\\*'):
        img = Image.open(file).resize(size)
        if img.mode != 'RGB':
            img = img.convert('RGB')  # convert grayscale images to RGB
        inputs.append(np.asarray(img))
        outputs.append(1)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    perm = np.random.permutation(len(inputs))
    inputs = inputs[perm]
    outputs = outputs[perm]

    return inputs, outputs, labels
