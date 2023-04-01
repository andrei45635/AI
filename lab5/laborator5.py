from lab5.classification import classificationV1, getRealPredictedLabels, classificationV2
from lab5.regression import regression
from lab5.utils import readFromCSV

if __name__ == '__main__':
    data = readFromCSV('data/sport.csv')
    data.pop(0)
    mae, rmse = regression(data)
    print("Mean Absolute Error: ", mae)
    print("Root Mean Square Error: ", rmse)

    flowers = readFromCSV('data/flowers1.csv')
    real, predicted = getRealPredictedLabels(flowers)
    cm, acc, precision, rc = classificationV1(real, predicted, list(set(real)))
    print('Confusion Matrix: ', cm)
    print('Accuracy: ', acc)
    print('Precision: ', precision)
    print('Recall: ', rc)

    accV2, precisionPos, precisionNeg, recallPos, recallNeg = classificationV2(real, predicted, list(set(real)))
    print('Accuracy: ', accV2)
    print('Positive precision: ', precisionPos)
    print('Negative precision: ', precisionNeg)
    print('Positive recall: ', recallPos)
    print('Negative recall: ', recallNeg)
