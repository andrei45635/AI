import random
from random import randint, uniform
from lab4.utils.permutation import generatePermutation, generatePermsFixed
from lab4.utils.binary_vect import generateBinaryVector


class Chromosome:
    def __init__(self, params=None):
        self.__params = params
        self.__repres = generatePermsFixed(self.__params['noNodes'], self.__params['start'], self.__params['end'])
        # self.__repres = generatePermutation(self.__params['noNodes'])
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
        """
        r = randint(0, len(self.__repres) - 1)
        new_representation = []
        for i in range(r):
            new_representation.append(self.__repres[i])
        for i in range(r, len(self.__repres)):
            new_representation.append(c.__repres[i])
        offspring = Chromosome(c.__params)
        offspring.representation = new_representation
        return offspring
        """

        # order XO
        pos1 = randint(1, self.__params['noNodes'] - 1)
        pos2 = randint(1, self.__params['noNodes'] - 1)
        if pos2 < pos1:
            pos1, pos2 = pos2, pos1
        k = 1

        newrepres = self.__repres[pos1: pos2]
        newrepres.insert(0, 0)
        for el in c.__repres[pos2:] + c.__repres[:pos2]:
            if el not in newrepres:
                if len(newrepres) < self.__params['noNodes'] - pos1:
                    newrepres.append(el)
                else:
                    newrepres.insert(k, el)
                    k += 1
        newrepres.remove(self.__params['start'])
        newrepres.remove(self.__params['end'])
        newrepres.insert(0, self.__params['start'])
        newrepres.insert(len(newrepres), self.__params['end'])
        offspring = Chromosome(self.__params)
        offspring.repres = newrepres
        return offspring

        # ===============================================
        # newrepres = []
        # vect = generateBinaryVector(len(self.__repres))
        # for i in range(len(self.__repres)):
        # if vect[i] == 0:
        #   newrepres.append(self.__repres[i])
        # else:
        # newrepres.append(c.__repres[i])
        # offspring = Chromosome(self.__params)
        # offspring.__repres = newrepres
        # return offspring
        # ===============================================

        # ========================================
        # rnd = randint(0, len(self.__repres) - 1)
        # newrepres = []
        # for i in range(rnd):
        # newrepres.append(self.__repres[i])
        # for i in range(rnd, len(self.__repres)):
        # newrepres.append(self.__repres[i])
        # offspring = Chromosome(c.__params)
        # offspring.__repres = newrepres
        # return offspring
        # ========================================

    def mutation(self):

        self.repres.remove(self.__params['start'])
        self.repres.remove(self.__params['end'])
        # pos1 = randint(0, len(self.__repres) - 1)
        # pos2 = randint(0, len(self.__repres) - 1)
        # self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]
        pos = randint(0, len(self.__repres) - 1)
        rando = randint(self.__params['min'], self.__params['max'])
        if rando != self.__params['start'] and rando != self.__params['end']:
            self.__repres[pos] = rando
        self.__repres = list(dict.fromkeys(self.__repres))
        self.__repres.insert(0, self.__params['start'])
        self.__repres.insert(len(self.__repres), self.__params['end'])

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
