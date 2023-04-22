from numpy import min, max


def normalisation(gdpTrainInputs, freedomTrainInputs, happinessTrainOutputs, gdpTestInputs, freedomTestInputs,
                  happinessTestOutputs, trainInputs, validationOutputs):
    for i in range(len(trainInputs)):
        gdpTrainInputs[i] = (gdpTrainInputs[i] - min(gdpTrainInputs)) / (max(gdpTrainInputs) - min(gdpTrainInputs))
        freedomTrainInputs[i] = (freedomTrainInputs[i] - min(freedomTrainInputs)) / (max(freedomTrainInputs) - min(freedomTrainInputs))
        happinessTrainOutputs[i] = (happinessTrainOutputs[i] - min(happinessTrainOutputs)) / (max(happinessTrainOutputs) - min(happinessTrainOutputs))

    for i in range(len(validationOutputs)):
        gdpTestInputs[i] = (gdpTestInputs[i] - min(gdpTestInputs)) / (max(gdpTestInputs) - min(gdpTestInputs))
        freedomTestInputs[i] = (freedomTestInputs[i] - min(freedomTestInputs)) / (max(freedomTestInputs) - min(freedomTestInputs))
        happinessTestOutputs[i] = (happinessTestOutputs[i] - min(happinessTestOutputs)) / (max(happinessTestOutputs) - min(happinessTestOutputs))
