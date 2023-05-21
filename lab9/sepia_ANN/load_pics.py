import glob
import os
import cv2

from PIL import Image

import numpy as np


def procImg(path):
    img = np.asarray(Image.open(path).resize((360,360)).convert('RGB'))
    processed = []
    for i in img:
        processed += list(i)
    return np.ravel(processed) / 255.0


def loadPics():
    inputs, outputs = [], []
    labels = ['!Sepia', 'Sepia']

    for file in glob.iglob('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul 4\\AI\\lab2\\lab9\\sepia_ANN\\data\\original\\*'):
        inputs.append(procImg(file))
        outputs.append(0)

    for file in glob.iglob('C:\\Users\\GIGABYTE\\OneDrive\\Desktop\\Facultate\\Semestrul 4\\AI\\lab2\\lab9\\sepia_ANN\\data\\sepia\\*'):
        inputs.append(procImg(file))
        outputs.append(1)

    return inputs, outputs, labels


def loadPictures(directory, img_size):
    outputNames = ['original', 'sepia']
    data = []
    for label in outputNames:
        path = os.path.join(directory, label)
        class_num = outputNames.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
