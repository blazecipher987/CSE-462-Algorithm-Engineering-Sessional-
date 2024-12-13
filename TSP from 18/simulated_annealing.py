# %%
import random
import math
import networkx as nx
from itertools import pairwise

# random.seed(0)

# %%
class SimulateAnnealing:
    def __init__(self, 
                 graph: nx.Graph, 
                 initial_temperature: float=1000, 
                 cooling_rate: float=0.01,
                 max_iterations: int=10,
                 initial_solution: list = None):
        self.graph = graph

        self.current_solution = initial_solution # [1, 2, 3, 4, 1]
        self.best_solution = initial_solution

        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

        self.trace = []
    
    def random_initial_solution(self):
        '''
        Generate a random initial solution
        '''
        if self.current_solution is None:
            nodes = self.graph.number_of_nodes()

            curr_node = random.choice(nodes)
            self.current_solution = [curr_node]

            available_nodes = set(nodes).remove(curr_node)
            while available_nodes:
                next_node = min(available_nodes, key=lambda x: self.graph.get_edge_data(curr_node, x)['weight'])
                available_nodes.remove(next_node)
                self.current_solution.append(next_node)
                curr_node = next_node

            self.best_solution = self.current_solution


    def cost(self, solution: list):
        '''
        Calculate the cost of the solution
        '''
        cost = sum(self.graph.get_edge_data(u, v)['weight'] for u, v in pairwise(solution))
    
        return cost
    
    def generate_new_solution(self):
        '''
        Generate a new solution by reversing a random subsequence of the current solution
        '''
        nodes = self.graph.number_of_nodes()

        neighbour = list(self.current_solution)[:-1]
        l = random.randint(2, nodes-2)
        i = random.randint(0, nodes-l)
        neighbour[i : (i + l)] = reversed(neighbour[i : (i + l)])

        neighbour.append(neighbour[0])

        return neighbour
    
    def acceptance_test(self, candidate_solution: list):
        '''
        Acceptance test for the new solution
        '''
        candidate_cost = self.cost(candidate_solution)
        current_cost = self.cost(self.current_solution)
        if candidate_cost < current_cost:
            self.current_solution = candidate_solution
            if candidate_cost < self.cost(self.best_solution):
                self.best_solution = candidate_solution
        else:
            if random.random() < math.exp((current_cost - candidate_cost) / self.temperature):
                self.current_solution = candidate_solution

    def update_temperature(self):
        self.temperature -= self.cooling_rate * self.temperature

    def run(self):
        # generate initial solution
        self.random_initial_solution()
        self.trace = [(self.current_solution, self.cost(self.current_solution))]

        curr_iter = 0
        while curr_iter <= self.max_iterations and self.temperature > 0:
            # generate new solution
            candidate_solution = self.generate_new_solution()

            # evaluate new solution
            self.acceptance_test(candidate_solution)

            # update trace
            self.trace.append((self.current_solution, self.cost(self.current_solution)))

            # decrease temperature
            self.update_temperature()

            curr_iter += 1

        return self.best_solution, self.cost(self.best_solution)
    
    def get_trace(self):
        return self.trace