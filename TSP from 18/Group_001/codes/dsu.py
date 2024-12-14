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
