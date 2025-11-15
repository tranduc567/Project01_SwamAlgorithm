import time
import numpy as np
from src.abc import ABC  # Đảm bảo bạn import đúng class ABC

def test_abc_tsp(distances, n_runs=5, num_bees=30, limit=50, max_iter=200):
    results = []
    for _ in range(n_runs):
        abc = ABC(num_bees=num_bees,
                  limit=limit,
                  max_iter=max_iter,
                  mode="discrete",
                  distance_matrix=distances)
        start = time.time()
        best_path, best_dist = abc.run()
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
    print("Testing ABC on TSP")
    abc_results = test_abc_tsp(distances)
    print_results_table("ABC results (dist, time):", abc_results)

if __name__ == "__main__":
    main()
