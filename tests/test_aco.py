import time
import numpy as np
from src.aco import AntColony


def test_aco_tsp(distances, n_runs=5, alpha=1, beta=2, decay=0.1):
    results = []
    for _ in range(n_runs):
        aco = AntColony(distances, n_ants=10, n_best=5, n_iterations=200,
                        decay=decay, alpha=alpha, beta=beta)
        start = time.time()
        best_path, best_dist = aco.run()
        end = time.time()
        results.append((best_dist, end - start))
    return results


def sensitivity_analysis(distances, param_name, param_values, n_runs=5):
    print(f"\nParameter Sensitivity Analysis on {param_name} (ACO)")
    print(f"{param_name:<7} | {'Avg Best Dist':<13} | {'Avg Time (s)':<11}")
    print("-" * 40)
    for val in param_values:
        best_dists = []
        times = []
        for _ in range(n_runs):
            # Thiết lập tham số mặc định
            alpha, beta, decay = 1, 2, 0.1
            if param_name == 'alpha':
                alpha = val
            elif param_name == 'beta':
                beta = val
            elif param_name == 'decay':
                decay = val
            
            aco = AntColony(distances, n_ants=10, n_best=5, n_iterations=50,
                            decay=decay, alpha=alpha, beta=beta)
            start = time.time()
            best_path, best_dist = aco.run()
            end = time.time()
            best_dists.append(best_dist)
            times.append(end - start)
        avg_dist = np.mean(best_dists)
        avg_time = np.mean(times)
        print(f"{val:<7} | {avg_dist:<13.4f} | {avg_time:<11.4f}")

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
    print("Testing ACO on TSP")
    aco_results = test_aco_tsp(distances)
    print_results_table("ACO results (dist, time):", aco_results)
    # --- Phân tích độ nhạy tham số alpha ---
    sensitivity_analysis(distances, 'alpha', [0.5, 1, 2, 5])
    # --- Phân tích độ nhạy tham số beta ---
    sensitivity_analysis(distances, 'beta', [0.5, 1, 2, 5])
    # --- Phân tích độ nhạy tham số decay ---
    sensitivity_analysis(distances, 'decay', [0.01, 0.05, 0.1, 0.3])

if __name__ == "__main__":
   # Load file distances.npy
    main()
