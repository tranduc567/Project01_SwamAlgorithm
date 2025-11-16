import numpy as np
import time
from utils import get_converge_epoch
import matplotlib.pyplot as plt
from ackley import ackley


def hill_climbing_continuous(obj_func, n_dims, lb, ub, epochs=1000):
    """Liên tục: tối ưu hàm số"""
    current = np.random.uniform(lb, ub, n_dims)
    best = current.copy()
    best_val = obj_func(best)
    history = []

    for _ in range(epochs):
        neighbor = best + np.random.uniform(-0.1, 0.1, n_dims)
        neighbor = np.clip(neighbor, lb, ub)
        val = obj_func(neighbor)
        if val < best_val:
            best = neighbor
            best_val = val
        history.append([best, best_val])
    return best, best_val, history


def hc_test(n_dims):
    lb = [-5] * n_dims
    ub = [5] * n_dims

    start = time.time()
    best_sol, best_fit, history = hill_climbing_continuous(obj_func=ackley, lb=lb, ub=ub, n_dims=n_dims, epochs=200)
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_fit)

    print("=== KẾT QUẢ Hill Climbing ===")
    print("Best position:", best_sol)
    print("Best fitness :", best_fit)
    print("Run time: ", run_time)
    print("Converge epoch: ", converge_epoch)

    # Lấy fitness qua các epoch
    fitness_over_time = [h[1] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(fitness_over_time)
    plt.xlabel("Epoch")
    plt.ylabel("Best fitness")
    plt.title("Hill Climbing Convergence on Ackley Function")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    hc_test(1)