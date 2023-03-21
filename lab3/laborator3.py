import warnings
from collections import Counter
import networkx as nx
import networkx.algorithms.community as nx_comm
from random import seed

from lab3.GA import GA
from lab3.chromosome import Chromosome


def readGML(fileName):
    g = nx.read_gml(fileName, label=None)
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
# param = readGML('data/football/football.gml')
# param = readGML('data/karate/karate.gml')
# param = readGML('data/krebs/krebs.gml')
# param = readGML('data/adjnoun/adjnoun.gml')
# param = readGML('data/lesmis/lesmis.gml')
# param = readGML('data/netscience/netscience.gml')

MIN = 0
MAX = param['noNodes']


def modularity(communities):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    m = param['edges']
    q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if communities[i] == communities[j]:
                q += (mat[i][j] - degrees[i] * degrees[j] / m)
    return abs(q * (1 / m))


def solveGA():
    generations = []

    seed(1)

    gaParam = {'popSize': 100, 'noGen': 1000}
    problParam = {'min': MIN, 'max': MAX, 'function': modularity, 'noNodes': MAX}

    ga = GA(gaParam, problParam)
    ga.initialistion()
    ga.eval()

    allComms = []
    allFitnesses = []

    bestestComs = 0
    bestestChromosome = Chromosome(problParam)
    bestestChromosome.fitness = 0
    bestestRepr = []

    for g in range(gaParam['noGen']):
        generations.append(g)

        bestChromosome = ga.bestChromosome()

        communities_dict = {}
        for i in range(len(bestChromosome.repres)):
            if bestChromosome.repres[i] in communities_dict:
                communities_dict[bestChromosome.repres[i]].append(i)
            else:
                communities_dict[bestChromosome.repres[i]] = [i]

        allComms.append(len(communities_dict))
        allFitnesses.append(bestChromosome.fitness)

        if bestestChromosome.fitness < bestChromosome.fitness:
            bestestChromosome = bestChromosome
            bestestRepr = bestChromosome.repres
            bestestComs = len(communities_dict)

        ga.oneGeneration()
        # ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()

        print('Generation: ' + str(g) + ' communities: ' + str(
            len(Counter(bestChromosome.repres).items())) + ' best solution in generation = ' + str(
            bestChromosome.repres) + ' fitness = ' + str(bestChromosome.fitness))

    print("Best chromosome fitness: ", bestestChromosome.fitness)

    print("Fitness evolution of the best chromosome: ")
    print(allFitnesses)

    print("Communities for the best chromosome: " + str(bestestComs))

    print("Community evolution: ")
    print(allComms)

    print('Index communities for the best chromosome: ')
    for i in range(0, len(bestestRepr)):
        print(str(i) + ": " + str(bestestRepr[i]))


if __name__ == "__main__":
    solveGA()
    warnings.simplefilter('ignore')
