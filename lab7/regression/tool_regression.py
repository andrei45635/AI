from sklearn import linear_model


def univariableRegressionTool(trainGDP, trainHappiness, testGDP):
    x = [[el] for el in trainGDP]
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=1000)
    regressor.fit(x, trainHappiness)
    w0, w1 = regressor.intercept_[0], regressor.coef_[0]
    model = f"the learnt model: f(x) = {w0} + {w1} * x"
    computedOutputs = regressor.predict([[x] for x in testGDP])
    return model, computedOutputs


def multivariableRegressionTool(trainInputs, trainOutputs, validationInputs):
    x = [[el1, el2] for el1, el2 in trainInputs]
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=1000)
    regressor.fit(x, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    model = f"the learnt model: f(x1, x2) = {w0} + {w1} * x1 + {w2} * x2"
    computedOutputs = regressor.predict(validationInputs)
    return model, computedOutputs
