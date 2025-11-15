import time
import numpy as np
from src.fa import FireflyAlgorithm  # chắc chắn import đúng

def test_fa_tsp(distances, n_runs=5, num_fireflies=25, max_iter=100, alpha=0.5, beta0=1.0, gamma=1.0):
    results = []
    for _ in range(n_runs):
        fa = FireflyAlgorithm(num_fireflies=num_fireflies,
                              max_iter=max_iter,
                              alpha=alpha,
                              beta0=beta0,
                              gamma=gamma,
                              mode="discrete",
                              distance_matrix=distances)
        start = time.time()
        best_path, best_dist = fa.run()
        end = time.time()
        results.append((best_dist, end - start))
    return results

def parameter_sensitivity_analysis_fa(distances, param_name, param_values, n_runs=5):
    print(f"\nParameter Sensitivity Analysis on {param_name} (FA)")
    print(f"{param_name:<7} | {'Avg Best Dist':<13} | {'Avg Time (s)':<11}")
    print("-" * 40)
    for val in param_values:
        best_dists = []
        times = []
        for _ in range(n_runs):
            # Thiết lập tham số mặc định
            alpha, beta0, gamma = 0.5, 1.0, 1.0
            if param_name == 'alpha':
                alpha = val
            elif param_name == 'beta0':
                beta0 = val
            elif param_name == 'gamma':
                gamma = val

            fa = FireflyAlgorithm(num_fireflies=25,
                                  max_iter=100,
                                  alpha=alpha,
                                  beta0=beta0,
                                  gamma=gamma,
                                  mode="discrete",
                                  distance_matrix=distances)
            start = time.time()
            best_path, best_dist = fa.run()
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
    print("Testing Firefly Algorithm on TSP")
    fa_results = test_fa_tsp(distances)
    print_results_table("FA results (dist, time):", fa_results)

    # Phân tích độ nhạy tham số alpha
    parameter_sensitivity_analysis_fa(distances, 'alpha', [0.1, 0.3, 0.5, 0.7, 1.0])

    # Phân tích độ nhạy tham số beta0
    parameter_sensitivity_analysis_fa(distances, 'beta0', [0.1, 0.5, 1.0, 2.0, 5.0])

    # Phân tích độ nhạy tham số gamma
    parameter_sensitivity_analysis_fa(distances, 'gamma', [0.1, 0.5, 1.0, 2.0, 5.0])

if __name__ == "__main__":
    main()
