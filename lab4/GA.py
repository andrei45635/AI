from random import randint
from lab4.chromosome import Chromosome


class GA:
    def __init__(self, params=None, problParams=None):
        self.__params = params
        self.__problParams = problParams
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialistion(self):
        for _ in range(0, self.__params['popSize']):
            c = Chromosome(self.__problParams)
            self.__population.append(c)

    def eval(self):
        for c in self.__population:
            c.fitness = self.__problParams['function'](self.__problParams['net']['mat'], c.repres)

    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness < best.fitness:
                best = c
        return best

    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness > best.fitness:
                best = c
        return best

    def selection(self):
        pos1 = randint(0, self.__params['popSize'] - 1)
        pos2 = randint(0, self.__params['popSize'] - 1)
        if self.__population[pos1].fitness < self.__population[pos2].fitness:
            return pos1
        else:
            return pos2

    def oneGeneration(self):
        newPop = []
        for _ in range(0, self.__params['popSize']):
            father = self.__population[self.selection()]
            mother = self.__population[self.selection()]
            offspring = father.crossover(mother)
            offspring.mutation()
            newPop.append(offspring)
        self.__population = newPop
        self.eval()

    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        for _ in range(0, self.__params['popSize'] - 1):
            father = self.__population[self.selection()]
            mother = self.__population[self.selection()]
            offspring = father.crossover(mother)
            offspring.mutation()
            newPop.append(offspring)
        self.__population = newPop
        self.eval()

    def oneGenerationSteadyState(self):
        for _ in range(0, self.__params['popSize'] - 1):
            father = self.__population[self.selection()]
            mother = self.__population[self.selection()]
            offspring = father.crossover(mother)
            offspring.mutation()
            offspring.fitness = self.__problParams['function'](self.__problParams['net']['mat'], offspring.repres)
            worst = self.worstChromosome()
            if offspring.fitness < worst.fitness:
                worst = offspring
