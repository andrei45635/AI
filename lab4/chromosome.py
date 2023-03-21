from random import randint
from lab4.utils.permutation import generatePermutation, generatePermsFixed
from lab4.utils.binary_vect import generateBinaryVector


class Chromosome:
    def __init__(self, params=None):
        self.__params = params
        self.__repres = generatePermsFixed(self.__params['noNodes'], self.__params['start'], self.__params['end'])
        self.__fitness = 0.0

    @property
    def fitness(self):
        return self.__fitness

    @property
    def repres(self):
        return self.__repres

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    @repres.setter
    def repres(self, l=[]):
        self.__repres = l

    def crossover(self, c):
        rnd = randint(0, len(self.__repres) - 1)
        newrepres = []
        for i in range(rnd):
            newrepres.append(self.__repres[i])
        for i in range(rnd, len(self.__repres)):
            newrepres.append(self.__repres[i])
        offspring = Chromosome(c.__params)
        offspring.__repres = newrepres
        return offspring

    def experimental_crossover(self, c):
        newrepres = []
        vect = generateBinaryVector(len(self.__repres))
        for i in range(len(self.__repres)):
            if vect[i] == 0:
                newrepres.append(self.__repres[i])
            else:
                newrepres.append(c.__repres[i])
        offspring = Chromosome(self.__params)
        offspring.__repres = newrepres
        return offspring

    def mutation(self):
        pos1 = randint(0, len(self.__repres) - 1)
        pos2 = randint(0, len(self.__repres) - 1)
        self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
