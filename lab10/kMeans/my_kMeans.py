import random
from random import uniform

import numpy as np


def euclideanDistance(point, data) -> float:
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data) ** 2, axis=1))


class AndreiKMeans:
    def __init__(self, nClusters=8, maxIters=1000):
        self.__centroids = None
        self.__nClusters = nClusters
        self.__maxIters = maxIters

    def fit(self, trainInput):
        # initializing the centroids using the <<k-means++>> method
        # we select a random datapoint as the first and then the rest are initialized
        # with probabilities proportional to their distances to the first
        self.__centroids = [random.choice(trainInput)]
        for _ in range(self.__nClusters - 1):
            dists = np.sum([euclideanDistance(centroid, trainInput) for centroid in self.__centroids], axis=0)
            dists /= np.sum(dists)
            newCentroidIndex, = np.random.choice(range(len(trainInput)), size=1, p=dists)
            self.__centroids.append([trainInput[newCentroidIndex]])

        min_, max_ = np.min(trainInput, axis=0), np.max(trainInput, axis=0)  # select centroids randomly
        self.__centroids = [uniform(max_, min_) for _ in
                            range(self.__nClusters)]  # uniformly distribute them across the whole domain

        iteration = 0
        prevCentroids = None
        while np.not_equal(self.__centroids, prevCentroids).any() and iteration < self.__maxIters:
            # sort each datapoint and assign it to the nearest centroid
            sortedPoints = [[] for _ in range(self.__nClusters)]
            for x in trainInput:
                dists = euclideanDistance(x, self.__centroids)
                centroidIndex = np.argmin(dists)
                sortedPoints[centroidIndex].append(x)
            # push current centroids to previous centroids
            # reassign centroids as mean of the points in them
            prevCentroids = self.__centroids
            self.__centroids = [np.mean(cluster, axis=0) for cluster in sortedPoints]
            for i, centroid in enumerate(self.__centroids):
                # this is to catch any errors from centroids having no points
                if np.isnan(centroid).any():
                    self.__centroids[i] = prevCentroids[i]
                iteration += 1

    def evaluate(self, train):
        centroids, centroidsIndexes = [], []
        for x in train:
            dists = euclideanDistance(x, self.__centroids)
            centroidIndex = np.argmin(dists)
            centroids.append(self.__centroids[centroidIndex])
            centroidsIndexes.append(centroidIndex)
        return centroids, centroidsIndexes
