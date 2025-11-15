import numpy as np
import random

class FireflyAlgorithm:
    def __init__(self,
                 num_fireflies=25,
                 max_iter=100,
                 alpha=0.5,        # độ rung lắc (randomness)
                 beta0=1.0,        # cường độ hấp dẫn ban đầu
                 gamma=1.0,        # hệ số suy giảm hấp dẫn theo khoảng cách
                 mode="continuous",
                 lower_bound=-5,
                 upper_bound=5,
                 dim=10,
                 distance_matrix=None):
        
        self.num_fireflies = num_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.mode = mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.distance_matrix = distance_matrix

        # Khởi tạo quần thể
        self.fireflies = []
        self.fitness = []

    def tsp_distance(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.distance_matrix[path[i], path[i + 1]]
        d += self.distance_matrix[path[-1], path[0]]
        return d

    def continuous_fitness(self, x):
        # Hàm mục tiêu ví dụ Sphere function
        return np.sum(x ** 2)

    def evaluate(self, solution):
        if self.mode == "discrete":
            return self.tsp_distance(solution)
        else:
            return self.continuous_fitness(solution)

    def init_solution(self):
        if self.mode == "discrete":
            return np.random.permutation(len(self.distance_matrix))
        else:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def move_firefly(self, xi, xj, beta):
        if self.mode == "discrete":
            # Di chuyển kiểu swap 2 vị trí gần giống với làm mềm permutation
            xi_new = xi.copy()
            # Lấy vị trí khác nhau giữa xi và xj
            diff_positions = [idx for idx in range(len(xi)) if xi[idx] != xj[idx]]
            if len(diff_positions) > 1:
                # Swap 2 vị trí để xi tiến gần xj
                i, j = random.sample(diff_positions, 2)
                xi_new[i], xi_new[j] = xi_new[j], xi_new[i]
            return xi_new
        else:
            # Liên tục: move theo công thức FA chuẩn
            rand = self.alpha * (np.random.rand(self.dim) - 0.5)
            return xi + beta * (xj - xi) + rand

    def distance(self, xi, xj):
        if self.mode == "discrete":
            # Khoảng cách Hamming giữa 2 permutation
            return np.sum(xi != xj)
        else:
            return np.linalg.norm(xi - xj)

    def run(self):
        # Khởi tạo quần thể
        self.fireflies = [self.init_solution() for _ in range(self.num_fireflies)]
        self.fitness = [self.evaluate(f) for f in self.fireflies]

        best_idx = np.argmin(self.fitness)
        best_sol = self.fireflies[best_idx].copy()
        best_val = self.fitness[best_idx]

        for _ in range(self.max_iter):
            for i in range(self.num_fireflies):
                for j in range(self.num_fireflies):
                    if self.fitness[j] < self.fitness[i]:
                        rij = self.distance(self.fireflies[i], self.fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * (rij ** 2))
                        new_sol = self.move_firefly(self.fireflies[i], self.fireflies[j], beta)
                        new_fit = self.evaluate(new_sol)

                        if new_fit < self.fitness[i]:
                            self.fireflies[i] = new_sol
                            self.fitness[i] = new_fit

                            if new_fit < best_val:
                                best_val = new_fit
                                best_sol = new_sol.copy()

        return best_sol, best_val
