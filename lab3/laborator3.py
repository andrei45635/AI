import warnings
from collections import Counter
import networkx as nx
from random import seed

from lab3.GA import GA
from lab3.chromosome import Chromosome


def readGML(fileName):
    g = nx.read_gml(fileName)
    mat = nx.to_numpy_array(g)
    net = {'noNodes': g.number_of_nodes(), 'edges': g.number_of_edges(), 'mat': mat}

    degrees = []
    noEdges = 0
    for i in range(0, net['noNodes']):
        d = 0
        for j in range(0, net['noNodes']):
            if mat.item((i, j)) == 1:
                d += 1
            if j > i:
                noEdges += mat.item((i, j))
        degrees.append(d)
    net['degrees'] = degrees
    return net


param = readGML('data/dolphins/dolphins.gml')
MIN = 0
MAX = param['noNodes']


def modularity(communities):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    edges = param['edges']
    m = 2 * edges
    q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if communities[i] == communities[j]:
                q += (mat.item((i, j)) - degrees[i] * degrees[j] / m)
    return q * 1 / m


def solveGA():
    generations = []

    seed(1)

    gaParam = {'popSize': 100, 'noGen': 100}
    problParam = {'min': MIN, 'max': MAX, 'function': modularity, 'noNodes': MAX}

    ga = GA(gaParam, problParam)
    ga.initialistion()
    ga.eval()

    bestestChromosome = Chromosome(problParam)
    for g in range(gaParam['noGen']):
        generations.append(g)

        ga.oneGeneration()
        # ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()

        bestChromosome = ga.bestChromosome()
        if bestestChromosome.fitness < bestChromosome.fitness:
            bestestChromosome = bestChromosome

        print('Generation: ' + str(g) + ' nr comunitati: ' + str(len(Counter(bestChromosome.repres).items())) + ' best solution in generation = ' + str(bestChromosome.repres) + ' fitness = ' + str(bestChromosome.fitness))

    print('\n')
    print("Communities: " + str(len(Counter(bestestChromosome.repres).keys())))
    print("Best solution: " + str(bestestChromosome.repres))
    print("Best fitness: " + str(bestestChromosome.fitness))


if __name__ == "__main__":
    solveGA()
    warnings.simplefilter('ignore')
