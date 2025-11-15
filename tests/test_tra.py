import time
import numpy as np
from src.traditional.hill_climbing import hill_climbing_tsp
from src.traditional.simulated_annealing import simulated_annealing_tsp
from src.traditional.genetic_algorithm import genetic_algorithm_tsp


def test_hill_climbing_tsp(distances, n_runs=5):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_path, best_dist = hill_climbing_tsp(distances)
        end = time.time()
        results.append((best_dist, end - start))
    return results

def test_simulated_annealing_tsp(distances, n_runs=5):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_path, best_dist = simulated_annealing_tsp(distances)
        end = time.time()
        results.append((best_dist, end - start))
    return results

def test_genetic_algorithm_tsp(distances, n_runs=5):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_path, best_dist = genetic_algorithm_tsp(distances)
        end = time.time()
        results.append((best_dist, end - start))
    return results

def print_results_table(name, results):
    print(f"\n{name}")
    print("-" * 40)
    print(f"{'Run':>3} | {'Distance':>12} | {'Time (s)':>10}")
    print("-" * 40)
    for i, (dist, t) in enumerate(results, 1):
        print(f"{i:3} | {dist:12.4f} | {t:10.4f}")
    print("-" * 40)

def main():
    distances = np.load("data/distances.npy")
    print("\nTesting Hill Climbing on TSP")
    hc_results = test_hill_climbing_tsp(distances)
    print_results_table("Hill Climbing results (dist, time):", hc_results)

    print("\nTesting Simulated Annealing on TSP")
    sa_results = test_simulated_annealing_tsp(distances)
    print_results_table("Simulated Annealing results (dist, time):", sa_results)

    print("\nTesting Genetic Algorithm on TSP")
    ga_results = test_genetic_algorithm_tsp(distances)
    print_results_table("Genetic Algorithm results (dist, time):", ga_results)

if __name__ == "__main__":
   # Load file distances.npy
    main()
