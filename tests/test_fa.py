import matplotlib.pyplot as plt
from src.ackley import ackley
from src.FA import FA
import time
from src.utils import get_converge_epoch


def fa_test(n_dims, max_sparks, p_a, p_b, max_ea, m_sparks):
    lb = [-5] * n_dims
    ub = [5] * n_dims

    fa = FA(
        obj_func=ackley,
        lb=lb,
        ub=ub,
        n_dims=n_dims,
        pop_size=50,
        epochs=200,
        max_sparks=max_sparks,
        p_a=p_a,
        p_b=p_b,
        max_ea=max_ea,
        m_sparks=m_sparks
    )

    # ---- Chạy tối ưu ----
    start = time.time()
    best_pos, best_fit, history = fa.solve()
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_fit)

    print("=== KẾT QUẢ FA ===")
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
    plt.title("FA Convergence on Ackley Function")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    fa_test(n_dims=2, max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50)