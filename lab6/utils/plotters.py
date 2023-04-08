from matplotlib import pyplot as plt


def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def plotGDP(gdp, happiness):
    plt.plot(gdp, happiness, 'ro')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.title('GDP capita vs. happiness')
    plt.show()


def plotFreedom(freedom, happiness):
    plt.plot(freedom, happiness, 'ro')
    plt.xlabel('Freedom')
    plt.ylabel('happiness')
    plt.title('Freedom vs. happiness')
    plt.show()


def plotAll(gdp, freedom, happiness):
    ax = plt.axes(projection='3d')
    ax.plot3D(gdp, freedom, happiness, 'ro')
    ax.set_xlabel('GDP')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('Happiness')
    plt.show()


def plotSplitData(train, validation):
    ax = plt.axes(projection = '3d')
    ax.plot3D([tr[0] for tr in train[0]], [tr[1] for tr in train[0]], train[1], 'ro', label = 'training data')
    ax.plot3D([val[0] for val in validation[0]], [val[1] for val in validation[0]], validation[1], 'g^', label = 'validation data')
    plt.legend()
    plt.show()
