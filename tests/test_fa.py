import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Dùng để vẽ 3D
from src.fa import FireflyAlgorithm
from src.utils import ackley_function_2d  # Hàm Ackley 2D cho surface plot
from src.utils import plot_ackley_surface  # Hàm Ackley 2D cho surface plot

sns.set(style="whitegrid")

def test_fa_continuous(n_runs=5, **kwargs):
    results = []
    histories = []
    for run in range(n_runs):
        fa = FireflyAlgorithm(mode="continuous", **kwargs)
        start = time.time()
        best_sol, best_fit, history = fa.run()
        end = time.time()
        results.append((best_fit, end - start))
        histories.append(history)
        print(f"Run {run+1}/{n_runs} done: Best fitness = {best_fit:.6f}, Time = {end-start:.4f}s")
    return results, histories

def print_results_table(name, results):
    print(f"\n{name}")
    print("-" * 40)
    print(f"{'Run':>3} | {'Best fitness':>12} | {'Time (s)':>10}")
    print("-" * 40)
    for i, (fit, t) in enumerate(results, 1):
        print(f"{i:3} | {fit:12.6f} | {t:10.4f}")
    print("-" * 40)

def plot_convergence(histories, title):
    plt.figure(figsize=(8,5))
    for h in histories:
        plt.plot(h, alpha=0.3)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

def parameter_sensitivity_continuous(param_name, param_values, fixed_params, n_runs=5):
    """
    Thử thay đổi tham số param_name trong param_values, chạy FA trên Ackley function,
    vẽ biểu đồ fitness trung bình so với giá trị tham số.
    """
    avg_fitness = []
    for val in param_values:
        params = fixed_params.copy()
        params[param_name] = val
        results, _ = test_fa_continuous(n_runs=n_runs, **params)
        mean_fit = np.mean([r[0] for r in results])
        avg_fitness.append(mean_fit)
        print(f"{param_name} = {val:.3f} -> Avg best fitness: {mean_fit:.6f}")

    plt.figure(figsize=(8,5))
    sns.lineplot(x=param_values, y=avg_fitness, marker="o")
    plt.title(f"Parameter Sensitivity on {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Average Best Fitness")
    plt.grid(True)
    plt.show()


def main():
    # --- Test continuous ---
    # print("Testing Firefly Algorithm on continuous problem (Ackley function)")
    # fixed_params = {
    #     'num_fireflies': 30,
    #     'max_iter': 200,
    #     'alpha': 0.3,
    #     'beta0': 1.0,
    #     'gamma': 1.0,
    #     'lower_bound': -32.768,
    #     'upper_bound': 32.768,
    #     'dim': 10,
    # }
    # cont_results, cont_histories = test_fa_continuous(n_runs=5, **fixed_params)
    # print_results_table("Continuous problem results (Ackley):", cont_results)
    # plot_convergence(cont_histories, "FA Convergence on Ackley Function")

    # # --- Parameter Sensitivity Analysis ---
    # parameter_sensitivity_continuous('alpha', [0.1, 0.3, 0.5, 0.7, 1.0], fixed_params)
    # parameter_sensitivity_continuous('beta0', [0.1, 0.5, 1.0, 2.0, 5.0], fixed_params)
    # parameter_sensitivity_continuous('gamma', [0.1, 0.5, 1.0, 2.0, 5.0], fixed_params)

    # --- 3D Surface Plot ---
    plot_ackley_surface()

if __name__ == "__main__":
    main()
