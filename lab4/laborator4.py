import os

from lab4.GA import GA
from lab4.chromosome import Chromosome
from lab4.utils.fitness import fitness


def readFile(filePath):
    net = {}
    costs = []

    with open(filePath, 'r') as f:
        noNodes = int(f.readline())
        for i in range(0, noNodes):
            line = []
            buffer = f.readline().strip().split(',')
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

    gaParam = {'popSize': 100, 'noGen': 100}
    problParam = {'min': MIN, 'max': MAX, 'function': fitness, 'noNodes': MAX, 'start': net['start'], 'end': net['end'], 'net': net}

    ga = GA(gaParam, problParam)
    ga.initialistion()
    ga.eval()

    bestestChromosome = Chromosome(problParam)
    bestestFitness = 999999999

    for g in range(gaParam['noGen']):
        generations.append(g)

        ga.oneGeneration()
        # ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()

        bestChromosome = ga.bestChromosome()
        if bestestChromosome.fitness < bestChromosome.fitness:
            bestestChromosome = bestChromosome

        print('Generation: ' + str(g) + ' best solution: ' + str(bestChromosome.repres) + ' fitness: ' + str(bestChromosome.fitness))


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'easy_tsp1.txt')
    nt = readFile(filePath)
    print(nt)
    solveGA(nt)



