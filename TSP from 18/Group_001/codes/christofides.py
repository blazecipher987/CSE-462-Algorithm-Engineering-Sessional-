import networkx as nx
import tsplib95
from mst import MST
import utils

file =  open('data/gr17.tsp')
graph = tsplib95.read(file).get_graph()

mst = MST()
mst_graph = mst.get_mst_k(graph, k = 20)
utils.plot_graph(mst_graph, 'mst.png')

# odd_nodes = MST.get_odd_degree_nodes(mst_graph)
# subgraph = nx.subgraph(graph, odd_nodes)
# utils.plot_graph(subgraph, name='subgraph.png')

# new_graph = nx.Graph()
# for e in graph.edges:
#     new_graph.add_edge(e[0], e[1], weight=-graph.get_edge_data(e[0], e[1])["weight"])

# set_matching = nx.max_weight_matching(new_graph, maxcardinality=True)
# print(set_matching)

# matching_graph = nx.Graph()
# for m in set_matching:
#     matching_graph.add_edge(m[0], m[1], weight=graph.get_edge_data(m[0], m[1])["weight"])

# # return matching_graph
# utils.plot_graph(matching_graph, name="min_matching.png")