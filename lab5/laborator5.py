from lab5.classification import classificationV1, getRealPredictedLabels, classificationV2, lossClassification, lossV2, \
    multiClassLoss, multiTargetLoss
from lab5.regression import regression
from lab5.utils import readFromCSV, readTxt, readFile, readMultiClass, readMultiTarget

if __name__ == '__main__':

    sports = readFromCSV('data/sport.csv')
    sports.pop(0)
    mae, rmse = regression(sports)
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
    
    probBinary = 'data/probabilities-binary.txt'
    trueBinary = 'data/true-binary.txt'
    r1, p1 = readFile(probBinary)
    real, predicted = readTxt(trueBinary)
    loss = lossClassification(real, predicted)
    loss1 = lossV2(real, predicted, r1, p1)
    print('Loss for the binary classification', probBinary, trueBinary, 'is', loss, loss1)

    probMultiClass = 'data/probabilities-multi-class.txt'
    trueMultiClass = 'data/true-multi-class.txt'
    npx = readMultiClass(probMultiClass)
    npx1 = readMultiClass(trueMultiClass)
    mulitClassLoss = multiClassLoss(npx, npx1)
    print('Multi class loss for the files', probMultiClass, trueMultiClass, 'is', mulitClassLoss)

    probMultiTarget = 'data/probabilities-multi-target.txt'
    trueMultiTarget = 'data/true-multi-target.txt'
    npx2 = readMultiTarget(probMultiTarget)
    npx3 = readMultiTarget(trueMultiTarget)
    multiTgLoss = multiTargetLoss(npx2, npx3)
    print('Multi target loss for the files', probMultiTarget, trueMultiTarget, 'is', multiTgLoss)
