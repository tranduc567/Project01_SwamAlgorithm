import matplotlib.pyplot as plt
from src.ackley import ackley
from src.CSA import CSA
import time
from src.utils import get_converge_epoch


def csa_test(n_dims, p_a):
    lb = [-5] * n_dims
    ub = [5] * n_dims

    csa = CSA(
        obj_func=ackley,
        lb=lb,
        ub=ub,
        n_dims=n_dims,
        pop_size=50,
        epochs=200,
        p_a=p_a
    )

    # ---- Chạy tối ưu ----
    start = time.time()
    best_pos, best_fit, history = csa.solve()
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_fit)

    print("=== KẾT QUẢ CSA ===")
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
    plt.title("CSA Convergence on Ackley Function")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csa_test(n_dims=2, p_a=0.03)