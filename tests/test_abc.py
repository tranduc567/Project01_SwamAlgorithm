import matplotlib.pyplot as plt
from src.ackley import ackley
from src.ABC import ABC
import time
from src.utils import get_converge_epoch


def abc_test(n_dims, limits):
    lb = [-5] * n_dims
    ub = [5] * n_dims

    abc = ABC(
        obj_func=ackley,
        lb=lb,
        ub=ub,
        n_dims=n_dims,
        pop_size=50,
        epochs=200,
        limits=limits
    )

    # ---- Chạy tối ưu ----
    start = time.time()
    best_pos, best_fit, history = abc.solve()
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_fit)

    print("=== KẾT QUẢ PSO ===")
    print("Best position:", best_pos)
    print("Best fitness :", best_fit)
    print("Run time: ", run_time)
    print("Converge epoch: ", converge_epoch)

    # Lấy fitness qua các epoch
    fitness_over_time = [h[1] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(fitness_over_time)
    plt.xlabel("Epoch")
    plt.ylabel("Best fitness")
    plt.title("ABC Convergence on Ackley Function")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    abc_test(n_dims=2, limits=10)