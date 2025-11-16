import numpy as np
import time
import matplotlib.pyplot as plt
from src.utils import get_converge_epoch
from src.traditional.ga import ga_tsp


def ga_test(n_cities, p_t, seed=42):
    rng = np.random.default_rng(seed)
    distance_matrix = rng.integers(low=1, high=100, size=(n_cities, n_cities)).astype(float)

    # Đặt đường chéo = 0
    np.fill_diagonal(distance_matrix, 0)

    # Đảm bảo ma trận đối xứng (d[i,j] = d[j,i])
    i_lower = np.tril_indices(n_cities, -1)
    distance_matrix[i_lower] = distance_matrix.T[i_lower]

    start = time.time()
    best_solution, best_value, history = ga_tsp(distance_matrix=distance_matrix, pop_size=50, epoch=200, mutation_rate=p_t)
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_value)


    print(f"\n=== KẾT QUẢ GA {n_cities} thành phố ===")
    print("Best tour:", best_solution)
    print("Best distance:", best_value)
    print("Run time: ", run_time)
    print("Converge epoch: ", converge_epoch)

    # Vẽ hội tụ
    # Lấy fitness từ history
    fitness_over_time = [item[1] for item in history]  # item[1] = best_value

    plt.plot(fitness_over_time, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Best value")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    n_cities=20
    seed = 123
    ga_test(n_cities=n_cities, p_t=0.3, seed=seed)
