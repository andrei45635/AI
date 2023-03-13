import os
import networkx as nx
import matplotlib.pyplot as plt
import warnings


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


def modularity(gr, cmty):
    k_i = 0
    edge_qs = list(nx.edge_betweenness_centrality(gr).items())
    m = gr.number_of_edges()
    m_comm = gr.subgraph(cmty).number_of_edges()
    for node in cmty:
        k_i += gr.degree(node)
    q = ((m_comm / m) - ((k_i / 2 * m) ** 2))
    return max(edge_qs, key=lambda item: item[1])[0]


def colorCommunities(gr):
    community = [1] * g.number_of_nodes()
    color = 0
    for comm in nx.connected_components(gr):
        color += 1
        for node in comm:
            community[node] = color
    return community


def greedyCommunitiesDetection(gr, comms):
    while len(list(nx.connected_components(gr))) < comms:
        sink, source = modularity(gr, [])
        gr.remove_edge(sink, source)

    color_communities = colorCommunities(gr)
    return color_communities


def plotNetwork(gr, cms):
    if not cms:
        cms = [0 for _ in range(0, gr.number_of_edges())]
    pos = nx.spring_layout(gr, k=0.3, iterations=20)
    nx.draw_networkx(gr, pos, node_size=600, node_color=cms, arrows=False, with_labels=True)
    plt.show()


if __name__ == '__main__':
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'footballEdges.txt')
    nt = readNet(filePath)
    adj_mat = nx.adjacency_matrix(nt)
    g = nx.Graph(adj_mat)
    coms = greedyCommunitiesDetection(g, 2)
    print(coms)
    plotNetwork(g, coms)

warnings.simplefilter('ignore')
