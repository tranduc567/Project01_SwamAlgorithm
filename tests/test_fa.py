import time
import numpy as np
import matplotlib.pyplot as plt
from src.fa import FireflyAlgorithm  # Đảm bảo import đúng

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

def test_fa_discrete(distances, n_runs=5, **kwargs):
    results = []
    histories = []
    for run in range(n_runs):
        fa = FireflyAlgorithm(mode="discrete", distance_matrix=distances, **kwargs)
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

def main():
    # --- Test continuous (Ackley function) ---
    print("Testing Firefly Algorithm on continuous problem (Ackley function)")
    cont_results, cont_histories = test_fa_continuous(
        n_runs=5,
        num_fireflies=30,
        max_iter=200,
        alpha=0.3,
        beta0=1.0,
        gamma=1.0,
        lower_bound=-32.768,
        upper_bound=32.768,
        dim=10,
    )
    print_results_table("Continuous problem results (Ackley):", cont_results)
    plot_convergence(cont_histories, "FA Convergence on Ackley Function")

    # --- Test discrete (TSP) ---
    print("Testing Firefly Algorithm on discrete problem (TSP)")
    distances = np.load("data/distances.npy")
    disc_results, disc_histories = test_fa_discrete(
        distances,
        n_runs=5,
        num_fireflies=30,
        max_iter=200,
        alpha=0.3,
        beta0=1.0,
        gamma=0.1,
    )
    print_results_table("Discrete TSP results:", disc_results)
    plot_convergence(disc_histories, "FA Convergence on TSP")

if __name__ == "__main__":
    main()
