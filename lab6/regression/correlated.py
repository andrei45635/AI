import pandas as pd


def getCorrelationMatrix(fileName):
    df = pd.read_csv(fileName)
    return df.corr()
