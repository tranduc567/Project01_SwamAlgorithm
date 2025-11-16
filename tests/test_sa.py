import time
from src.utils import get_converge_epoch
import matplotlib.pyplot as plt
from src.traditional import sa_continuous
from src.ackley import ackley


def sa_test(n_dims):
    lb = [-5] * n_dims
    ub = [5] * n_dims

    start = time.time()
    best_sol, best_fit, history = sa_continuous(obj_func=ackley, lb=lb, ub=ub, n_dims=n_dims, epochs=200)
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_fit)

    print("=== KẾT QUẢ Simulated Annealing ===")
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
    plt.title("Simulated Annealing Convergence on Ackley Function")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    sa_test(4)