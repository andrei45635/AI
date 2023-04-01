import os

from matplotlib import pyplot as plt
from tsplib95 import loaders
from lab4.GA import GA
from lab4.chromosome1 import Chromosome
# from lab4.chromosome import Chromosome
from lab4.utils.fitness import fitness, fitCyclic


def readTSP(filePath):
    tsp = loaders.load(filePath)
    print(tsp)


def readFile(filePath):
    net = {}
    costs = []

    with open(filePath, 'r') as f:
        noNodes = int(f.readline())
        for i in range(0, noNodes):
            line = []
            buffer = f.readline().strip().split(' ')
            for cost in buffer:
                line.append(int(cost))
            costs.append(line)
        start = int(f.readline())
        end = int(f.readline())

    net['noNodes'] = noNodes
    net['mat'] = costs
    net['start'] = start
    net['end'] = end
    return net


def solveGA(net):
    MIN = 0
    MAX = net['noNodes']
    generations = []

    gaParam = {'popSize': 100, 'noGen': 25000}
    # problParam = {'min': MIN, 'max': MAX, 'function': fitness, 'noNodes': MAX, 'start': net['start'], 'end': net['end'], 'net': net}
    problParam = {'min': MIN, 'max': MAX, 'function': fitness, 'noNodes': MAX, 'net': net}

    ga = GA(gaParam, problParam)
    ga.initialistion()
    ga.eval()

    allFitnesses = []
    allBestFitnesses = []
    allAvgFitnesses = []
    bestsFits = []

    bestestChromosome = Chromosome(problParam)
    bestestChromosome.fitness = 99999999

    for g in range(gaParam['noGen']):
        generations.append(g)

        bestChromosome = ga.bestChromosome()
        last_pops = ga.population

        # ga.oneGeneration()
        ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()

        if bestestChromosome.fitness > bestChromosome.fitness:
            bestestChromosome.repres = bestChromosome.repres
            bestestChromosome.fitness = bestChromosome.fitness

        for gen in last_pops:
            if gen.fitness == bestestChromosome.fitness and gen not in bestsFits:
                bestsFits.append(gen)

        allPotentialSolutionsX = [c.repres for c in ga.population]
        allPotentialSolutionsY = [c.fitness for c in ga.population]
        bestSolX = ga.bestChromosome().repres
        bestSolY = ga.bestChromosome().fitness
        allBestFitnesses.append(bestSolY)
        allAvgFitnesses.append(sum(allPotentialSolutionsY) / len(allPotentialSolutionsY))
        allFitnesses.append(bestChromosome.fitness)

        print('Generation: ' + str(g) + ' best solution: ' + str(bestChromosome.repres) + ' fitness: ' + str(bestChromosome.fitness))

    print('Best Chromosome: ', bestestChromosome.repres, 'fitness: ', bestestChromosome.fitness)
    print('Fitness evolution: ', allFitnesses)
    print('Best solution: ', bestsFits)
    plt.ioff()
    print(len(generations), len(allBestFitnesses))
    best = plt.plot(generations, allBestFitnesses, 'ro', label='best')
    mean = plt.plot(generations, allAvgFitnesses, 'bo', label='mean')
    plt.show()
    #a
    print(bestestChromosome)


if __name__ == '__main__':
    crtDir = os.getcwd()
    path = os.path.join(crtDir, 'data', 'hard_tsp1.txt')
    nt = readFile(path)
    # readTSP(path)
    solveGA(nt)




