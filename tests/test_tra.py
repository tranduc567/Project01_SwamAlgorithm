import time
import numpy as np
from src.traditional.hill_climbing import hill_climbing_tsp, hill_climbing_continuous
from src.traditional.simulated_annealing import simulated_annealing_tsp, simulated_annealing_continuous
from src.traditional.genetic_algorithm import genetic_algorithm_tsp, genetic_algorithm_continuous
from src.utils import ackley_function

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

def test_hill_climbing_continuous(n_runs=5, dim=10, lower_bound=-32.768, upper_bound=32.768):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_sol, best_fit = hill_climbing_continuous(ackley_function, dim, lower_bound, upper_bound)
        end = time.time()
        results.append((best_fit, end - start))
    return results

def test_simulated_annealing_continuous(n_runs=5, dim=10, lower_bound=-32.768, upper_bound=32.768):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_sol, best_fit = simulated_annealing_continuous(ackley_function, dim, lower_bound, upper_bound)
        end = time.time()
        results.append((best_fit, end - start))
    return results

def test_genetic_algorithm_continuous(n_runs=5, dim=10, lower_bound=-32.768, upper_bound=32.768):
    results = []
    for _ in range(n_runs):
        start = time.time()
        best_sol, best_fit = genetic_algorithm_continuous(ackley_function, dim, lower_bound, upper_bound)
        end = time.time()
        results.append((best_fit, end - start))
    return results

def print_results_table(name, results):
    print(f"\n{name}")
    print("-" * 40)
    print(f"{'Run':>3} | {'Best fitness':>12} | {'Time (s)':>10}")
    print("-" * 40)
    for i, (fit, t) in enumerate(results, 1):
        print(f"{i:3} | {fit:12.6f} | {t:10.4f}")
    print("-" * 40)

def main():
    distances = np.load("data/distances.npy")

    # Test các thuật toán trên bài toán TSP
    print("\nTesting Hill Climbing on TSP")
    hc_results = test_hill_climbing_tsp(distances)
    print_results_table("Hill Climbing results (dist, time):", hc_results)

    print("\nTesting Simulated Annealing on TSP")
    sa_results = test_simulated_annealing_tsp(distances)
    print_results_table("Simulated Annealing results (dist, time):", sa_results)

    print("\nTesting Genetic Algorithm on TSP")
    ga_results = test_genetic_algorithm_tsp(distances)
    print_results_table("Genetic Algorithm results (dist, time):", ga_results)

    # Test các thuật toán trên bài toán tối ưu hóa liên tục (Ackley)
    print("\nTesting Hill Climbing on Ackley function")
    hc_cont_results = test_hill_climbing_continuous()
    print_results_table("Hill Climbing continuous results (fitness, time):", hc_cont_results)

    print("\nTesting Simulated Annealing on Ackley function")
    sa_cont_results = test_simulated_annealing_continuous()
    print_results_table("Simulated Annealing continuous results (fitness, time):", sa_cont_results)

    print("\nTesting Genetic Algorithm on Ackley function")
    ga_cont_results = test_genetic_algorithm_continuous()
    print_results_table("Genetic Algorithm continuous results (fitness, time):", ga_cont_results)

if __name__ == "__main__":
    main()
