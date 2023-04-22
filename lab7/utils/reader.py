import pandas as pd


def loadAsDFUnivar(filePath):
    df = pd.read_csv(filePath)
    df.fillna(df.mean(), inplace=True)
    subset = df[['Economy..GDP.per.Capita.', 'Happiness.Score']]

    gdp = [subset.iat[i, 0] for i in range(len(subset))]
    happiness = [subset.iat[i, 1] for i in range(len(subset))]

    return gdp, happiness


def loadAsDFMultivar(filePath):
    df = pd.read_csv(filePath)
    df.fillna(df.mean(), inplace=True)
    subset = df[['Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score']]

    gdp = [subset.iat[i, 0] for i in range(len(subset))]
    freedom = [subset.iat[i, 1] for i in range(len(subset))]
    happiness = [subset.iat[i, 2] for i in range(len(subset))]

    return gdp, freedom, happiness
