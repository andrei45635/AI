import numpy as np


def divideData(gdp, freedom, happiness):
    np.random.seed(5)
    indexes = [i for i in range(len(gdp))]
    trainSample = np.random.choice(indexes, int(0.8 * len(gdp)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    trainInputs = [[gdp[i], freedom[i]] for i in trainSample]
    trainOutputs = [happiness[i] for i in trainSample]

    validationInputs = [[gdp[i], freedom[i]] for i in validationSample]
    validationOutputs = [happiness[i] for i in validationSample]

    return trainInputs, trainOutputs, validationInputs, validationOutputs
