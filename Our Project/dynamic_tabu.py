import tsplib95
import os
import numpy as np
from itertools import permutations
from collections import deque
import matplotlib.pyplot as plt

class TabuSearchTSP:
    def __init__(self, filename, initial_tabu_size=5, max_tabu_size=20, max_iter=500):
        self.problem = tsplib95.load(filename)
        self.cities = list(self.problem.get_nodes())
        self.distances = self._calculate_distances()
        self.tabu_list = deque(maxlen=initial_tabu_size)
        self.initial_tabu_size = initial_tabu_size
        self.max_tabu_size = max_tabu_size
        self.max_iter = max_iter
        self.performance = [] 
        self.tabu_sizes = []   

    def _calculate_distances(self):
        n = len(self.cities)
        distances = np.zeros((n, n))
        for i, city1 in enumerate(self.cities):
            for j, city2 in enumerate(self.cities):
                if i != j:
                    distances[i][j] = self.problem.get_weight(city1, city2)
        return distances

    def _calculate_tour_cost(self, tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.distances[tour[i] - 1][tour[i + 1] - 1]
        cost += self.distances[tour[-1] - 1][tour[0] - 1]
        return cost

    def _generate_neighborhood(self, tour):
        neighbors = []
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                neighbor = tour[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def _adjust_tabu_size(self, iteration, current_cost, best_cost):
        improvement_rate = abs(current_cost - best_cost) / max(1, best_cost)
        
        # Increase tabu size if improvement slows down
        if improvement_rate < 0.01 and len(self.tabu_list) < self.max_tabu_size:
            self.tabu_list = deque(self.tabu_list, maxlen=len(self.tabu_list) + 1)
        # Decrease tabu size if significant improvement is observed
        elif improvement_rate > 0.05 and len(self.tabu_list) > self.initial_tabu_size:
            self.tabu_list = deque(self.tabu_list, maxlen=len(self.tabu_list) - 1)


    def solve(self):
        current_tour = self.cities[:]
        np.random.shuffle(current_tour)
        current_cost = self._calculate_tour_cost(current_tour)
        best_tour = current_tour[:]
        best_cost = current_cost

        for iteration in range(self.max_iter):
            neighbors = self._generate_neighborhood(current_tour)
            neighbor_costs = [
                self._calculate_tour_cost(neighbor) for neighbor in neighbors
            ]

            for neighbor, cost in sorted(zip(neighbors, neighbor_costs), key=lambda x: x[1]):
                if neighbor not in self.tabu_list:
                    current_tour = neighbor
                    current_cost = cost
                    self.tabu_list.append(neighbor)
                    break

            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost

            # Adjust tabu size dynamically
            self._adjust_tabu_size(iteration, current_cost, best_cost)

            self.performance.append(best_cost)
            self.tabu_sizes.append(len(self.tabu_list))  # Record tabu list size
            print(f"Iteration {iteration + 1}: Best Cost = {best_cost}, Tabu Size = {len(self.tabu_list)}")

        return best_tour, best_cost

    def get_performance(self):
        return self.performance, self.tabu_sizes


def plot_results(results, output_folder="./plots"):
    os.makedirs(output_folder, exist_ok=True)

    for filename, (performance, tabu_sizes) in results.items():
        iterations = range(len(performance))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the performance (cost) on the primary y-axis
        ax1.plot(iterations, performance, label="Tour Cost", color="blue", linewidth=2)
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Tour Cost", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.grid()

        # Annotate graph with costs at specific iterations
        for i in range(0, len(performance), max(1, len(performance) // 10)):
            ax1.text(i, performance[i], f"{performance[i]:.2f}", fontsize=8, ha="right", color="blue")

        # Plot the tabu list size on the secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(iterations, tabu_sizes, label="Tabu List Size", color="orange", linestyle="--", linewidth=2)
        ax2.set_ylabel("Tabu List Size", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Title and legends
        plt.title(f"Performance on {filename}")
        fig.tight_layout()
        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        # Save the plot as a PNG file
        plot_filename = f"{filename}_performance.png".replace("/", "_")
        plt.savefig(os.path.join(output_folder, plot_filename), dpi=300)
        plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    data_folder = "./datasets"  # Folder containing .tsp files
    max_iters = 100  #
    max_files = 2
    results = {}
    for filename in sorted(os.listdir(data_folder))[:max_files]:
        if filename.endswith(".tsp"):
            filepath = os.path.join(data_folder, filename)
            print(f"\nRunning on dataset: {filename}")
            tsp_solver = TabuSearchTSP(filepath, max_iter=max_iters)
            _, cost = tsp_solver.solve()
            performance, tabu_sizes = tsp_solver.get_performance()
            results[filename] = (performance, tabu_sizes)
    plot_results(results)