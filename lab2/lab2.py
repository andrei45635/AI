import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

from networkx.algorithms import community


def readNetAdjList(fileName):
    edges = []
    f = open(fileName, "r")
    nodes = int(f.readline()) + 1
    net = {'nodes': nodes}
    line = f.readline()
    while line:
        edges.append(tuple(map(int, line.replace("\n", "").split())))
        line = f.readline()
    size = len(set([n for edge in edges for n in edge]))
    adj = [[0] * size for _ in range(size)]
    for i, j in edges:
        adj[i][j] = 1
    net['mat'] = adj
    f.close()
    return net


def readNet(fileName):
    f = open(fileName, "r")
    gr = nx.read_edgelist(f)
    f.close()
    return gr


def plotNetwork(ntwk, cms):
    g = nx.Graph(matrix=ntwk)
    nx.draw(g, with_labels=True)
    pos = nx.spring_layout(g)  # compute graph layout
    plt.figure(figsize=(4, 4))  # image is 8 x 8 inches
    nx.draw_networkx_nodes(g, pos, node_size=50, cmap=plt.get_cmap('viridis', 8), node_color=cms)
    nx.draw_networkx_edges(g, pos, alpha=0.3)
    plt.show()


def modularity(g, cmty):
    k_i = 0
    edge_qs = list(nx.edge_betweenness_centrality(g).items())
    m = g.number_of_edges()
    m_comm = g.subgraph(cmty).number_of_edges()
    for node in cmty:
        k_i += g.degree(node)
    q = ((m_comm / m) - ((k_i / 2 * m) ** 2))
    print(q / m)
    return max(edge_qs, key=lambda item: item[1])[0]


def greedyCommunitiesDetection(g, comms):
    adj = np.matrix(g['mat'])
    gr = nx.from_numpy_array(adj)

    while len(list(nx.connected_components(gr))) < comms:
        sink, source = modularity(gr, [])
        gr.remove_edge(sink, source)

    com = [1] * g['nodes']
    color = 0
    for comm in nx.connected_components(gr):
        color += 1
        for node in comm:
            com[node] = color
    return com


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'dolphinsEdges.txt')
    nt = readNet(filePath)
    network = readNetAdjList(filePath)
    coms = greedyCommunitiesDetection(network, 2)
    print(coms)
    plotNetwork(network, coms)

warnings.simplefilter('ignore')
