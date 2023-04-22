import numpy as np
import sklearn


def myUnivariableBatchGD(trainInputs, trainOutputs, testInputs, testOutputs, alpha, iters):
    x = np.c_[np.ones((len(trainInputs), 1)), trainInputs]
    y = np.array(trainOutputs).flatten()
    beta = np.array([0, 0])
    for _ in range(iters):
        hypothesis = x.dot(beta)
        loss = hypothesis - y
        gradient = np.dot(np.transpose(x), loss) / len(y)
        beta = beta - alpha * gradient
    computedOutputs = [beta[0] + testInputs[i] * beta[1] for i in range(len(testInputs))]
    model = f"the learnt model: f(x) = {beta[0]} + {beta[1]} * x"
    error = sklearn.metrics.mean_squared_error(testOutputs, computedOutputs)
    return model, error


def myMultivariableBatchGD(trainInputs, trainOutputs, alpha, iters):
    happiness_new = np.reshape(trainOutputs, (len(trainOutputs), 1))
    gdp = np.c_[np.ones((len(trainInputs), 1)), trainInputs]
    theta = np.random.randn(len(gdp[0]), 1) # Generate an initial value of vector θ from the original independent variables matrix
    for _ in range(iters):
        gradients = 2 / len(happiness_new) * np.dot(np.transpose(gdp), (np.dot(gdp, theta) - happiness_new))
        theta = theta - alpha * gradients
    model = f"the learnt model: f(x1, x2) = {theta[0][0]} + {theta[1][0]} * x1 + {theta[2][0]} * x2"
    return model

