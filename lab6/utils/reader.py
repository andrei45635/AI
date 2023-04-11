import csv

import numpy as np
import pandas as pd


def loadData(fileName, firstInputVar, secondInputVar, outputVar):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(firstInputVar)
    gdp = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedVariable = dataNames.index(secondInputVar)
    freedom = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVar)
    happiness = [float(data[i][selectedOutput]) for i in range(len(data))]

    return gdp, freedom, happiness


def loadAsDF(filePath):
    df = pd.read_csv(filePath)
    dataNames = []
    df.fillna(df.mean(), inplace=True)
    subset = df[['Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score']]

    gdp = [subset.iat[i, 0] for i in range(len(subset))]
    freedom = [subset.iat[i, 1] for i in range(len(subset))]
    happiness = [subset.iat[i, 2] for i in range(len(subset))]

    return gdp, freedom, happiness


def correlatedDF(filePath):
    threshold = 0.8
    df1 = pd.read_csv(filePath)
    df = df1[['Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score']]
    df.astype({'Freedom': int, 'Happiness.Score': int})
    cor = df.corr()
    corrm = np.corrcoef(df.transpose())
    corr = corrm - np.diagflat(corrm.diagonal())
    print("max corr:", corr.max(), ", min corr: ", corr.min())
    c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
    high_cor = c1[c1.values != 1]
    thresh = threshold
    print(high_cor[high_cor > thresh])
    correlated_features = set()
    for i in range(len(cor.columns)):
        for j in range(i):
            if abs(cor.iloc[i, j]) > threshold:
                colname = cor.columns[i]
                correlated_features.add(colname)
    print(correlated_features)
    gdp = [df.iat[i, 0] for i in range(len(df))]
    freedom = [df.iat[i, 1] for i in range(len(df))]
    happiness = [df.iat[i, 2] for i in range(len(df))]

    return correlated_features, gdp, freedom, happiness
