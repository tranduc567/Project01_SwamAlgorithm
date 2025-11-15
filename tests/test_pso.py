import time
import numpy as np
from src.pso import ParticleSwarmOptimization  # chắc chắn import đúng class PSO

def test_pso_tsp(distances, n_runs=5, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    results = []
    for _ in range(n_runs):
        pso = ParticleSwarmOptimization(
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
            mode="discrete",
            distance_matrix=distances
        )
        start = time.time()
        best_path, best_dist = pso.run()
        end = time.time()
        results.append((best_dist, end - start))
    return results

def parameter_sensitivity_analysis_pso(distances, param_name, param_values, n_runs=5):
    print(f"\nParameter Sensitivity Analysis on {param_name} (PSO)")
    print(f"{param_name:<7} | {'Avg Best Dist':<13} | {'Avg Time (s)':<11}")
    print("-" * 40)
    for val in param_values:
        best_dists = []
        times = []
        for _ in range(n_runs):
            # Thiết lập tham số mặc định
            w, c1, c2 = 0.7, 1.5, 1.5
            if param_name == 'w':
                w = val
            elif param_name == 'c1':
                c1 = val
            elif param_name == 'c2':
                c2 = val

            pso = ParticleSwarmOptimization(
                num_particles=30,
                max_iter=100,
                w=w,
                c1=c1,
                c2=c2,
                mode="discrete",
                distance_matrix=distances
            )
            start = time.time()
            best_path, best_dist = pso.run()
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
    print("Testing PSO on TSP")
    pso_results = test_pso_tsp(distances)
    print_results_table("PSO results (dist, time):", pso_results)

    # Phân tích độ nhạy tham số w (inertia weight)
    parameter_sensitivity_analysis_pso(distances, 'w', [0.4, 0.6, 0.7, 0.8, 1.0])

    # Phân tích độ nhạy tham số c1 (cognitive)
    parameter_sensitivity_analysis_pso(distances, 'c1', [0.5, 1.0, 1.5, 2.0, 2.5])

    # Phân tích độ nhạy tham số c2 (social)
    parameter_sensitivity_analysis_pso(distances, 'c2', [0.5, 1.0, 1.5, 2.0, 2.5])

if __name__ == "__main__":
    main()
