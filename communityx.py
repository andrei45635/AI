import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def E(G: nx.Graph):
    # returns the total number of edges in a graph
    
    return G.number_of_edges()

def sum_of_degrees(G: nx.Graph):
    # returns the sum of the degrees for each node in a graph
    
    degrees_sum = 0
    for node in G.nodes:
        degrees_sum += G.degree(node)
        
    return degrees_sum
        
def color_communities(G: nx.Graph):
    # returns a color map representing the colors for each nodes 
    # the color of each node is given by the community it is in

    _G = G      # copy of the initial graph 
    
    color = 0
    communities = nx.connected_components(_G)
    for community in communities:
        for node in community:
            G.nodes[node]["color"] = color
            
        color += 1
    
    color_map = []
    for node_data in _G.nodes(data=True):
        color = node_data[1]["color"]
        color_map.append(color)
        
    return color_map
        
def modularity(G: nx.Graph):
    # returns the modularity metric for a graph 
    # Q ∈ [-0.5, 1)
    
    Q = 0
    communities = nx.connected_components(G)
    for community in communities:
        G1 = G.subgraph(community)
        Q += E(G1) / E(G) - (sum_of_degrees(G1) / (2*E(G))) ** 2 
    
    return Q

def community_detection(G: nx.Graph):
    # performs a greedy community detection using the Girvan Newman algorithm and a metric represented 
    # by the modularity of the graph to know when to stop
    
    epsilon = np.finfo(np.float32).eps
    while modularity(G) <= 0.3 + epsilon:      
        edge_betweenness = nx.edge_betweenness_centrality(G).items()
        edge_to_remove = sorted(edge_betweenness, key=lambda item: item[1], reverse=True)[0][0]
        
        A, B = edge_to_remove[0], edge_to_remove[1]
        G.remove_edge(A, B)
    
    color_map = color_communities(G)
    return color_map

def show_graph(G: nx.Graph, color_map: list = []):
    # displays a graph to the screen 
    # if no color_map is given, all nodes will use the same color
    
    if color_map == []:
        color_map = [0 for i in range(0, G.number_of_nodes())]
    
    pos = nx.spring_layout(G, k=0.15, iterations=20) 
    nx.draw_networkx(G, pos, node_size=600, node_color=color_map, arrows=False, with_labels=True)    
    plt.show()

if __name__ == '__main__':
    path = "dolphins/dolphins.gml"
    
    G = nx.Graph()
    try:
        G = nx.read_gml(path)
    except nx.exception.NetworkxError:
        G = nx.read_gml(path, label=True)
    
    color_map = community_detection(G)
    
    show_graph(G, color_map)
    