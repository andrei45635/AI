from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer


def extractFeaturesBOW(trainInputs, testInputs):  # Bag of Words
    vectorizer = CountVectorizer()
    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.transform(testInputs)
    return trainFeatures.toarray(), testFeatures.toarray()


def extractFeaturesTFIDF(trainInputs, testInputs, maxFeats):  # TF IDF - word granularity
    vectorizer = TfidfVectorizer()
    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.transform(testInputs)
    return trainFeatures.toarray(), testFeatures.toarray()


def extractFeaturesHashing(trainInputs, testInputs, nFeats):  # Hashing - BOW with hash codes
    vectorizer = HashingVectorizer()
    trainFeatures = vectorizer.fit_transform(trainInputs)
    testFeatures = vectorizer.fit_transform(testInputs)
    return trainFeatures.toarray(), testFeatures.toarray()




