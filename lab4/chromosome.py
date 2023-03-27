import random
from random import randint, uniform
from lab4.utils.permutation import generatePermutation, generatePermsFixed
from lab4.utils.binary_vect import generateBinaryVector


class Chromosome:
    def __init__(self, params=None):
        self.__params = params
        # self.__repres = []
        self.__repres = generatePermsFixed(self.__params['noNodes'], self.__params['start'], self.__params['end'])
        # self.__repres = generatePermutation(self.__params['noNodes'])
        self.__fitness = 0.0
        # self._init_repres()

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
        if len(self.__repres) == 2 or len(c.__repres) == 2:
            off = Chromosome(self.__params)
            off.__repres = c.__repres
            return off

        sims = []
        for i in range(min(len(self.__repres), len(c.__repres))):
            if self.__repres[i] == c.__repres[i]:
                sims.append(self.__repres[i])

        cut_i = randint(0, len(sims) - 1)
        cut = sims[cut_i - 1]

        new_repres = self.__repres[:cut + 1]
        for i in range(cut + 1, len(c.__repres)):
            if c.__repres[i] not in new_repres:
                new_repres.append(c.__repres[i])

        offspring = Chromosome(self.__params)
        offspring.__repres = new_repres

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
        """
        if len(self.__repres) == 2:
            return

        selected = randint(1, len(self.__repres) - 2)
        before = self.__repres[selected - 1]
        after = self.__repres[selected + 1]
        used = self.__repres[:selected] + self.__repres[selected + 2:]
        new_repres = [before]
        while new_repres[-1] != after:
            new_city = randint(0, self.__params['noNodes'] - 1)
            while new_city in used:
                new_city = randint(0, self.__params['noNodes'] - 1)
            new_repres.append(new_city)
            used.append(new_city)
        self.__repres = self.__repres[:selected - 1] + new_repres + self.__repres[selected + 2:]
        """

        self.repres.remove(self.__params['start'])
        self.repres.remove(self.__params['end'])
        # pos1 = randint(0, len(self.__repres) - 1)
        # pos2 = randint(0, len(self.__repres) - 1)
        # self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]
        pos = randint(0, len(self.__repres) - 1)
        rando = randint(self.__params['min'], self.__params['max'])
        while rando == self.__params['start'] and rando == self.__params['end']:
            rando = randint(self.__params['min'], self.__params['max'])
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

    def _init_repres(self):
        self.__repres.append(self.__params["start"])
        while self.__repres[-1] != self.__params["end"]:
            newNode = randint(1, self.__params["noNodes"])
            while newNode in self.__repres:
                newNode = randint(1, self.__params["noNodes"])
            self.__repres.append(newNode)