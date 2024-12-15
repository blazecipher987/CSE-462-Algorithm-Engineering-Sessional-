import networkx as nx
import numpy as np
import heapq

class DSU:
    def __init__(self, nodes) -> None:
        self.parent : dict = {n : n for n in nodes}
        self.rank : dict = {n : 0 for n in nodes}

    def find_parent(self, node_id : int):
        if node_id == self.parent[node_id]:
            return node_id
        self.parent[node_id] = self.find_parent(self.parent[node_id])
        return self.parent[node_id]

    def union_sets(self, node1:int, node2:int):
        node1 = self.find_parent(node1)
        node2 = self.find_parent(node2)
        if (node1 != node2):
            if self.rank[node1] < self.rank[node2]:
                temp = node1
                node1 = node2
                node2 = temp
            self.parent[node2] = node1
            if self.rank[node1] == self.rank[node2]:
                self.rank[node1] += 1

class MST:
    def __init__(self) -> None:
        self.cost = 0

    def get_mst_prims_k(self, graph: nx.Graph, k: int) -> nx.Graph:
        """
        Probabilistic Prim's Algorithm O(ElogV)
        """
        if len(graph.nodes) == 0:
            return nx.Graph()

        # Start with an arbitrary node (e.g., the first node)
        start_node = list(graph.nodes)[0]

        pq = []

        visited = set()

        # MST graph
        mst_graph = nx.Graph()

        # Add all edges from the starting node to the priority queue
        for neighbor, edge_data in graph[start_node].items():
            weight = edge_data['weight']
            heapq.heappush(pq, (weight, start_node, neighbor))

        visited.add(start_node)
        self.cost = 0

        eps = 1e-9

        # While the MST does not span all nodes
        while pq and len(visited) < len(graph.nodes):
            # Extract edges from the priority queue
            edges = []
            for _ in range(min(k, len(pq))):
                edges.append(heapq.heappop(pq))

            # Normalize weights using softmax for probabilistic selection
            weights = np.array([edge[0] for edge in edges])
            weights = (weights - np.mean(weights)) / max(np.std(weights), eps)
            softmax_values = np.exp(-weights)
            probabilities = list(softmax_values / sum(softmax_values))

            # Select an edge probabilistically
            selected_index = np.random.choice(len(edges), p=probabilities)
            weight, u, v = edges[selected_index]

            # Return unselected edges back to the priority queue
            for i, edge in enumerate(edges):
                if i != selected_index:
                    heapq.heappush(pq, edge)

            if v in visited:
                continue

            # Add the edge to the MST
            mst_graph.add_weighted_edges_from([(u, v, weight)])
            self.cost += weight

            visited.add(v)

            # Add all edges from the new node to the priority queue
            for neighbor, edge_data in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_data['weight'], v, neighbor))

        return mst_graph

    def get_mst_prims(self, graph: nx.Graph) -> nx.Graph:
        """
        Prim's Algorithm O(ElogV)
        """
        if len(graph.nodes) == 0:
            return nx.Graph()

        # Start with an arbitrary node (e.g., the first node)
        start_node = list(graph.nodes)[0]

        pq = [] # Priority queue

        visited = set()

        mst_graph = nx.Graph()

        # Add all edges from the starting node to the priority queue
        for neighbor, edge_data in graph[start_node].items():
            weight = edge_data['weight']
            heapq.heappush(pq, (weight, start_node, neighbor))

        visited.add(start_node)
        self.cost = 0

        while pq and len(visited) < len(graph.nodes):
            # Get the smallest edge (weight, u, v)
            weight, u, v = heapq.heappop(pq)

            if v in visited:
                continue

            # Add the edge to the MST
            mst_graph.add_weighted_edges_from([(u, v, weight)])
            self.cost += weight

            visited.add(v)

            # Add all edges from the new node to the priority queue
            for neighbor, edge_data in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_data['weight'], v, neighbor))

        return mst_graph

    def get_mst_kruskal_k(self, graph: nx.Graph, k : int) -> nx.Graph:
        edges = graph.edges

        weighted_edges = {}
        for u, v in edges:
            weighted_edges[(u,v)] = int(graph.get_edge_data(u,v)["weight"])
        weighted_edges = sorted(weighted_edges.items(), key=lambda x: x[1])

        i = len(weighted_edges) - 1
        
        while (weighted_edges[i][1] == 0):
            i -= 1
        weighted_edges = weighted_edges[:i]

        disjoin_set = DSU(graph.nodes)
        mst_graph = nx.Graph()
        edge_count = 0
        edge_target = graph.number_of_nodes() - 1
        self.cost = 0
        eps = 1e-9

        
        start = 0
        end = (start + k) % len(weighted_edges)

        while edge_count != edge_target:
            l = len(weighted_edges)
            # print(start, end)
            if start > end:
                weights = np.array([e[1] for e in weighted_edges[start:]] + [e[1] for e in weighted_edges[:end]])
            else:
                weights = np.array([e[1] for e in weighted_edges[start:end]])
            weights = (weights - np.mean(weights)) / max(np.std(weights), eps)
            softmax_values = np.exp(-np.array(weights))
            probabilities = list(softmax_values / sum(softmax_values))
            selected_edge_index = (np.random.choice(np.arange(len(weights)), p=probabilities) + start) % l
            # print(selected_edge_index)
            selected_edge = weighted_edges[selected_edge_index]
            
            u, v = selected_edge[0]
            if disjoin_set.find_parent(u) != disjoin_set.find_parent(v):
                disjoin_set.union_sets(u, v)
                self.cost += selected_edge[1]
                mst_graph.add_weighted_edges_from([(u,v,selected_edge[1])])
                edge_count += 1
            
            del weighted_edges[selected_edge_index]

            # start = (end - 1) % l
            # start = 0
            if selected_edge_index != start:
                start = start + 1
            
            # start = (selected_edge_index - 1) % l
            end = (start + k) % l
            
        return mst_graph

    def get_mst_kruskal(self, graph: nx.Graph) -> nx.Graph:
        """
        Kruskal's Algorithm O(ElogE)
        """
        edges = graph.edges

        weighted_edges = {}
        for u, v in edges:
            weighted_edges[(u,v)] = int(graph.get_edge_data(u,v)["weight"])
        
        weighted_edges = sorted(weighted_edges.items(), key=lambda x: x[1])

        disjoint_set = DSU(graph.nodes) # Using Disjoint Set for the creation of MST
        mst_graph = nx.Graph()
        for edge, weight in weighted_edges:
            u, v = edge
            if disjoint_set.find_parent(u) != disjoint_set.find_parent(v):
                disjoint_set.union_sets(u, v)
                self.cost += weight
                mst_graph.add_weighted_edges_from([(u,v,weight)])

        return mst_graph

    def get_mst_cost(self):
        return self.cost

    def get_odd_degree_nodes(graph: nx.Graph) -> list:
        return [node for node, degree in graph.degree if degree % 2 == 1]


