import os
from sklearn.metrics import mean_squared_error

from lab6.regression.my_regression import my_regression
from lab6.regression.sk_learn_regression import sk_learnRegression
from lab6.utils.data_division import divideData
from lab6.utils.plotters import plotGDP, plotFreedom, plotDataHistogram, plotAll, plotSplitData
from lab6.utils.reader import loadAsDF, loadData

if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')

    gdp, freedom, happiness = loadAsDF(filePath)
    gdp1, freedom1, happiness1 = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
    mdl = my_regression(gdp, freedom, happiness)
    print('regression using the formula in the lab6 jupyter notebook:\n', mdl)

    trainInputs, trainOutputs, validationInputs, validationOutputs = divideData(gdp1, freedom1, happiness1)

    toolModel, regressor = sk_learnRegression(trainInputs, trainOutputs)
    computedValidationOutputs = regressor.predict([x for x in validationInputs])

    print('regression using sk-learn:\n', toolModel)

    error = 0.0
    for t1, t2 in zip(computedValidationOutputs, validationOutputs):
        error += (t1 - t2) ** 2
    error = error / len(validationOutputs)
    print("prediction error (manual): ", error)

    error = mean_squared_error(validationOutputs, computedValidationOutputs)
    print("prediction error (tool): ", error)

    plotDataHistogram(gdp, 'GDP')
    plotDataHistogram(freedom, 'Freedom')
    plotDataHistogram(happiness, 'Happiness')
    plotAll(gdp, freedom, happiness)
    plotGDP(gdp, happiness)
    plotFreedom(freedom, happiness)
    plotSplitData([trainInputs, trainOutputs], [validationInputs, validationOutputs])
