import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(graph : nx.Graph, name : str):
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=False)
    weights = {edge : str(graph.get_edge_data(edge[0], edge[1])["weight"]) for edge in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)
    plt.savefig(name)
    plt.close()
