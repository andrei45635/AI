from random import randint
from lab3.utils import generateNewValue


class Chromosome:
    def __init__(self, params=None):
        self.__params = params
        self.__repres = [generateNewValue(params['min'], params['max']) for _ in range(params['noNodes'])]
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

    def mutation(self):
        pos = randint(0, len(self.__repres) - 1)
        self.__repres[pos] = generateNewValue(self.__params['min'], self.__params['max'])

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
