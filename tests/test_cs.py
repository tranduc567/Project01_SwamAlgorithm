import time
import numpy as np
from src.cs import CuckooSearch 

def test_cs_tsp(distances, n_runs=5, num_nests=25, pa=0.25, max_iter=100):
    results = []
    for _ in range(n_runs):
        cs = CuckooSearch(num_nests=num_nests,
                          pa=pa,
                          max_iter=max_iter,
                          mode="discrete",
                          distance_matrix=distances)
        start = time.time()
        best_path, best_dist = cs.run()
        end = time.time()
        results.append((best_dist, end - start))
    return results

def parameter_sensitivity_analysis_cs(distances, param_name, param_values, n_runs=5):
    print(f"\nParameter Sensitivity Analysis on {param_name} (CS)")
    print(f"{param_name:<7} | {'Avg Best Dist':<13} | {'Avg Time (s)':<11}")
    print("-" * 40)
    for val in param_values:
        best_dists = []
        times = []
        for _ in range(n_runs):
            # Thiết lập tham số mặc định
            num_nests, pa, max_iter = 25, 0.25, 100
            if param_name == 'num_nests':
                num_nests = val
            elif param_name == 'pa':
                pa = val
            elif param_name == 'max_iter':
                max_iter = val

            cs = CuckooSearch(num_nests=num_nests,
                              pa=pa,
                              max_iter=max_iter,
                              mode="discrete",
                              distance_matrix=distances)
            start = time.time()
            best_path, best_dist = cs.run()
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
    print("Testing Cuckoo Search Algorithm on TSP")
    cs_results = test_cs_tsp(distances)
    print_results_table("CS results (dist, time):", cs_results)

    # Phân tích độ nhạy tham số num_nests
    parameter_sensitivity_analysis_cs(distances, 'num_nests', [10, 25, 50, 100])

    # Phân tích độ nhạy tham số pa
    parameter_sensitivity_analysis_cs(distances, 'pa', [0.1, 0.25, 0.5, 0.75])

    # Phân tích độ nhạy tham số max_iter
    parameter_sensitivity_analysis_cs(distances, 'max_iter', [50, 100, 200])

if __name__ == "__main__":
    main()
