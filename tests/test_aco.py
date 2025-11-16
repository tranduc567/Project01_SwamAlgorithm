import numpy as np
from src.ACO import ACO, ACOProblem
from typing import List
import matplotlib.pyplot as plt
import time
from src.utils import get_converge_epoch


class TSPProblem(ACOProblem):
    def __init__(self, distance_matrix: np.ndarray):
        n = distance_matrix.shape[0]
        super().__init__(n_decisions=n, objective=None)
        self.distance_matrix = distance_matrix
        self.solution = [None] * n
        self.current_index = 0

    def feasible_solution(self) -> List[int]:
        # các thành phố chưa đi
        return [i for i in range(self.n_decisions) if i not in self.solution[:self.current_index]]

    def make_decision(self, index: int, value: int):
        self.solution[index] = value
        self.current_index += 1

    def is_complete(self) -> bool:
        return self.current_index >= self.n_decisions

    def build(self) -> List[int]:
        return [int(x) for x in self.solution]


    def evaluate(self, solution: List[int]) -> float:
        # tổng quãng đường theo chu trình (quay về điểm đầu)
        dist = 0.0
        for i in range(len(solution)):
            dist += self.distance_matrix[solution[i-1], solution[i]]
        return dist



def test_aco(n_cities, alpha, beta, rho, seed=42):
    rng = np.random.default_rng(seed)
    distance_matrix = rng.integers(low=1, high=100, size=(n_cities, n_cities)).astype(float)

    # Đặt đường chéo = 0
    np.fill_diagonal(distance_matrix, 0)

    # Đảm bảo ma trận đối xứng (d[i,j] = d[j,i])
    i_lower = np.tril_indices(n_cities, -1)
    distance_matrix[i_lower] = distance_matrix.T[i_lower]

    problem = TSPProblem(distance_matrix)

    aco = ACO(problem=problem, n_ants=50, epochs=200, alpha=alpha, beta=beta, rho=rho)

    start = time.time()
    best_solution, best_value, history = aco.solve()
    run_time = time.time() - start
    converge_epoch = get_converge_epoch(history=history, best_fit=best_value)


    print(f"\n=== KẾT QUẢ ACO {n_cities} thành phố ===")
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
    plt.title("ACO Convergence")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    n_cities=20
    seed = 123
    test_aco(n_cities=n_cities, alpha=0.1, beta=0.2, rho=0.5, seed=seed)
