import tsplib95
import os
import numpy as np
from itertools import permutations
from collections import deque
import matplotlib.pyplot as plt

class TabuSearchTSP:
    def __init__(self, filename, tabu_size=10, max_iter=500):
        self.problem = tsplib95.load(filename)
        self.cities = list(self.problem.get_nodes())
        self.distances = self._calculate_distances()
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.performance = []  # Track cost at each iteration

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

    def solve(self):
        current_tour = self.cities[:]
        np.random.shuffle(current_tour)
        current_cost = self._calculate_tour_cost(current_tour)
        best_tour = current_tour[:]
        best_cost = current_cost
        tabu_list = deque(maxlen=self.tabu_size)

        for iteration in range(self.max_iter):
            neighbors = self._generate_neighborhood(current_tour)
            neighbor_costs = [
                self._calculate_tour_cost(neighbor) for neighbor in neighbors
            ]

            for neighbor, cost in sorted(zip(neighbors, neighbor_costs), key=lambda x: x[1]):
                if neighbor not in tabu_list:
                    current_tour = neighbor
                    current_cost = cost
                    tabu_list.append(neighbor)
                    break

            if current_cost < best_cost:
                best_tour = current_tour[:]
                best_cost = current_cost

            self.performance.append(best_cost)
            print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

        return best_tour, best_cost

    def get_performance(self):
        return self.performance

# Experimentation and Plotting
def run_experiments(data_folder, tabu_sizes, max_iters, max_files=5):
    results = {}
    tsp_files = [f for f in os.listdir(data_folder) if f.endswith(".tsp")][:max_files]
    
    for filename in tsp_files:
        filepath = os.path.join(data_folder, filename)
        print(f"\nRunning on dataset: {filename}")
        for tabu_size in tabu_sizes:
            for max_iter in max_iters:
                print(f"Tabu Size: {tabu_size}, Max Iter: {max_iter}")
                tsp_solver = TabuSearchTSP(filepath, tabu_size=tabu_size, max_iter=max_iter)
                _, cost = tsp_solver.solve()
                performance = tsp_solver.get_performance()
                results[(filename, tabu_size, max_iter)] = performance
    return results


def plot_results(results, output_folder="./plots"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for key, performance in results.items():
        dataset, tabu_size, max_iter = key
        label = f"{dataset} (Tabu={tabu_size}, Iter={max_iter})"
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(performance)), performance, label=label)
        
        # Add annotations for every 10th iteration
        for i in range(0, len(performance), 10):
            plt.annotate(
                f"{performance[i]:.2f}",
                (i, performance[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center"
            )

        plt.xlabel("Iterations")
        plt.ylabel("Tour Cost")
        plt.title(f"Performance on {dataset} (Tabu={tabu_size}, Iter={max_iter})")
        plt.legend()
        plt.grid()

        # Save the plot as a PNG file
        plot_filename = f"{dataset}_Tabu_{tabu_size}_Iter_{max_iter}.png".replace("/", "_")
        plt.savefig(os.path.join(output_folder, plot_filename), dpi=300)
        plt.close()  # Close the figure to free memory

# Main Execution
if __name__ == "__main__":
    data_folder = "./data"  # Folder containing .tsp files
    tabu_sizes = [10, 20]  # Experiment with different tabu list sizes
    max_iters = [100]  # Experiment with different iteration limits

    # Run experiments on only the first 10 .tsp files
    results = run_experiments(data_folder, tabu_sizes, max_iters, max_files=5)
    plot_results(results)
