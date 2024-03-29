from sklearn import linear_model


def sk_learnRegression(trainInputs, trainOutputs):
    x = [[el1, el2] for el1, el2 in trainInputs]
    regressor = linear_model.LinearRegression()
    regressor.fit(x, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    modelstr = 'the learnt model: f(x1, x2) = ' + str(w0) + ' + ' + str(w1) + ' * x1' + ' + ' + str(w2) + ' * x2'
    return modelstr, regressor
