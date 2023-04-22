import os

from sklearn.metrics import mean_squared_error

from lab7.regression.my_bgdRegression import myUnivariableBatchGD, myMultivariableBatchGD
from lab7.regression.normalisation import normalisation
from lab7.regression.tool_regression import univariableRegressionTool, multivariableRegressionTool
from lab7.utils.data_division import divideUnivariable, divideMultivariable
from lab7.utils.plotters import plotUnivariable, plotOutputsUnivariable, plotMultivariable
from lab7.utils.reader import loadAsDFUnivar, loadAsDFMultivar


def univariableTool(fp):
    gdp, happiness = loadAsDFUnivar(fp)
    trainGDP, trainHappiness, testGDP, testHappiness = divideUnivariable(gdp, happiness)
    model, computedOutputs = univariableRegressionTool(trainGDP, trainHappiness, testGDP)
    print(model)
    print("prediction error (tool): ", mean_squared_error(testHappiness, computedOutputs))
    plotUnivariable(trainGDP, trainHappiness, testGDP, testHappiness)
    plotOutputsUnivariable(testGDP, testHappiness, computedOutputs)


def multiVariableTool(fp):
    gdp, happiness, freedom = loadAsDFMultivar(fp)
    trainInputs, trainOutputs, validationInputs, validationOutputs, gdpTrainInputs, freedomTrainInputs, happinessTrainOutputs, gdpTestOutputs, freedomTestOutputs, happinessTestOutputs = divideMultivariable(gdp, freedom, happiness)
    normalisation(gdpTrainInputs, freedomTrainInputs, happinessTrainOutputs, gdpTestOutputs, freedomTestOutputs,
                  happinessTestOutputs, trainInputs, validationOutputs)
    model, computedOutputs = multivariableRegressionTool(trainInputs, trainOutputs, validationInputs)
    print(model)
    print("prediction error (tool): ", mean_squared_error(validationOutputs, computedOutputs))
    plotMultivariable(gdp, freedom, happiness)


def myUnivariable(fp):
    gdp, happiness = loadAsDFUnivar(fp)
    trainInputs, trainOutputs, testInputs, testOutputs = divideUnivariable(gdp, happiness)
    model, error = myUnivariableBatchGD(trainInputs, trainOutputs, testInputs, testOutputs, 0.01, 2000)
    print(model)
    print("Error: ", error)


def myMultivariable(fp):
    gdp, happiness, freedom = loadAsDFMultivar(fp)
    trainInputs, trainOutputs, validationInputs, validationOutputs, gdpTrainInputs, freedomTrainInputs, happinessTrainOutputs, gdpTestOutputs, freedomTestOutputs, happinessTestOutputs = divideMultivariable(
        gdp, freedom, happiness)
    normalisation(gdpTrainInputs, freedomTrainInputs, happinessTrainOutputs, gdpTestOutputs, freedomTestOutputs,
                  happinessTestOutputs, trainInputs, validationOutputs)
    model = myMultivariableBatchGD(trainInputs, trainOutputs, 0.01, 1000)
    print(model)


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'v1_world-happiness-report-2017.csv')
    print("Univariable BGD using sklearn: ")
    univariableTool(filePath)
    print("Multivariable BGD using sklearn: ")
    multiVariableTool(filePath)
    print("Univariable BGD using my own code: ")
    myUnivariable(filePath)
    print("Multivariable BGD using my own code: ")
    myMultivariable(filePath)
