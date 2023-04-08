import csv

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