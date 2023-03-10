import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

from networkx.algorithms import community


def readNet(fileName):
    f = open(fileName, "r")
    gr = nx.read_edgelist(f)
    f.close()
    return gr


def plotNetwork(ntwk):
    g = nx.Graph(ntwk)
    nx.draw_networkx(g, with_labels=True, node_color="c", edge_color="k", font_size="8")
    plt.axis('off')
    plt.draw()
    # plt.show(g)
    plt.savefig("graph1.pdf")


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'dolphinsEdges.txt')
    network = readNet(filePath)
    plotNetwork(network)

warnings.simplefilter('ignore')
