import networkx as nx
import tsplib95
from mst import MST
import utils

file =  open('data/ts225.tsp')
graph = tsplib95.read(file).get_graph()
utils.plot_graph(graph, "graph.png")

mst_lib : nx.Graph = nx.algorithms.minimum_spanning_tree(graph)
sm = 0
for e in mst_lib.edges:
    sm += mst_lib[e[0]][e[1]]['weight']
print(sm)
utils.plot_graph(mst_lib, 'mst_lib.png')

for k in range(10):
    mst = MST()
    tree = mst.get_mst_k(graph, k = 10)
    print(mst.get_mst_cost())
# utils.plot_graph(tree, 'mst.png') 

# tsp = nx.algorithms.approximation.traveling_salesman.christofides
# path = tsp(graph, tree=tree)

# print(tsp)