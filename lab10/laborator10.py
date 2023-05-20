import os

from sklearn.metrics import accuracy_score

from lab10.kMeans.evaluations import predictByTool, predictByMe, predictSupervised, predictHybrid
from lab10.utils.data_division import divideData
from lab10.utils.extracting_features import extractFeaturesTFIDF, extractFeaturesBOW, extractFeaturesHashing
from lab10.utils.readers import readReviews, readEmotions


def reviews():
    print('\nEmotions')
    fp = os.path.join(crtDir, 'data', 'reviews_mixed.csv')
    text, sentiment, labels = readReviews(fp)
    trainInputs, trainOutputs, testInputs, testOutputs = divideData(text, sentiment)
    trainFeatures, testFeatures = extractFeaturesBOW(trainInputs, testInputs)
    # trainFeatures, testFeatures = extractFeaturesTFIDF(trainInputs, testInputs, 150)
    # trainFeatures, testFeatures = extractFeaturesHashing(trainInputs, testInputs, 2 ** 10)
    computedOutputs = predictByTool(trainFeatures, testFeatures, labels, len(set(labels)))
    myComputedOutputs, centroids, computedIndexes = predictByMe(trainFeatures, testFeatures, labels, len(set(labels)))
    supervisedOutput = predictSupervised(trainFeatures, trainOutputs, testFeatures)
    hybridOutput, prevAcc = predictHybrid(trainFeatures, trainOutputs, testFeatures, testOutputs)
    inverseTestOutputs = ['negative' if elem == 'positive' else 'positive' for elem in testOutputs]
    accuracyByTool = accuracy_score(testOutputs, computedOutputs)
    accuracyByToolInverse = accuracy_score(inverseTestOutputs, computedOutputs)
    print('Accuracy score by tool:', max(accuracyByTool, accuracyByToolInverse))

    accuracyByMe = accuracy_score(testOutputs, myComputedOutputs)
    accuracyByMeInverse = accuracy_score(inverseTestOutputs, myComputedOutputs)
    print('Accuracy score by me:', max(accuracyByMe, accuracyByMeInverse))
    print('Accuracy score supervised:', accuracy_score(testOutputs, supervisedOutput))
    print('Accuracy score hybrid before KMeans:', prevAcc)
    print('Accuracy score hybrid after KMeans:', accuracy_score(testOutputs, hybridOutput))
    print('Output computed by tool:  ', computedOutputs)
    print('Output computed by me:    ', myComputedOutputs)
    print('Output for supervised:    ', list(supervisedOutput))
    print('Output for hybrid:        ', list(hybridOutput))
    print('Real output:              ', testOutputs)


def emails():
    print('\nEmails')
    fp = os.path.join(crtDir, 'data', 'spam.csv')
    emailText, emailType, labels = readEmotions(fp)
    trainInputs, trainOutputs, testInputs, testOutputs = divideData(emailText, emailType)
    trainFeatures, testFeatures = extractFeaturesBOW(trainInputs, testInputs)
    # trainFeatures, testFeatures = extractFeaturesTFIDF(trainInputs, testInputs, 150)
    # trainFeatures, testFeatures = extractFeaturesHashing(trainInputs, testInputs, 2 ** 10)
    computedOutputs = predictByTool(trainFeatures, testFeatures, labels, len(set(labels)))
    myComputedOutputs, centroids, computedIndexes = predictByMe(trainFeatures, testFeatures, labels, len(set(labels)))
    inverseTestOutputs = ['spam' if elem == 'ham' else 'ham' for elem in testOutputs]
    accuracyByTool = accuracy_score(testOutputs, computedOutputs)
    accuracyByToolInverse = accuracy_score(inverseTestOutputs, computedOutputs)
    print('Accuracy score by tool:', max(accuracyByTool, accuracyByToolInverse))
    accuracyByMe = accuracy_score(testOutputs, myComputedOutputs)
    accuracyByMeInverse = accuracy_score(inverseTestOutputs, myComputedOutputs)
    print('Accuracy score by me:', max(accuracyByMe, accuracyByMeInverse))
    print('Output computed by tool:  ', computedOutputs)
    print('Output computed by me:    ', myComputedOutputs)
    print('Real output:              ', testOutputs)


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'reviews_mixed.csv')
    reviews()
    emails()
