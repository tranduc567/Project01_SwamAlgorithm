import numpy as np
import random

class CuckooSearch:
    def __init__(self,
                 num_nests=25,
                 pa=0.25,             # Xác suất bị phát hiện và thay tổ (discovery rate)
                 max_iter=100,
                 mode="continuous",   # "discrete" or "continuous"
                 lower_bound=-5,
                 upper_bound=5,
                 dim=10,
                 distance_matrix=None):

        self.num_nests = num_nests
        self.pa = pa
        self.max_iter = max_iter
        self.mode = mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.distance_matrix = distance_matrix

        self.nests = []
        self.fitness = []

    # Tính khoảng cách hoặc fitness
    def tsp_distance(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.distance_matrix[path[i], path[i + 1]]
        d += self.distance_matrix[path[-1], path[0]]
        return d

    def continuous_fitness(self, x):
        # Hàm benchmark ví dụ Sphere
        return np.sum(x ** 2)

    def evaluate(self, solution):
        if self.mode == "discrete":
            return self.tsp_distance(solution)
        else:
            return self.continuous_fitness(solution)

    # Khởi tạo solution
    def init_solution(self):
        if self.mode == "discrete":
            return np.random.permutation(len(self.distance_matrix))
        else:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    # Lévy flight cho bước nhảy
    def levy_flight(self, size):
        # Sử dụng phân phối Lévy đơn giản
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # Cập nhật solution theo Lévy flight (cho continuous)
    def levy_flight_step(self, solution):
        step = self.levy_flight(len(solution))
        new_solution = solution + 0.01 * step * (solution - np.mean(solution))
        # Giới hạn trong biên
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        return new_solution

    # Đổi chỗ 2 vị trí (discrete neighbor move)
    def swap_two(self, solution):
        new_sol = solution.copy()
        i, j = random.sample(range(len(solution)), 2)
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        return new_sol

    # Tạo một new solution từ một nest (dùng Lévy flight hoặc swap)
    def get_cuckoo(self, current):
        if self.mode == "discrete":
            # Thay vì Lévy flight, dùng swap 2 vị trí
            return self.swap_two(current)
        else:
            return self.levy_flight_step(current)

    def run(self):
        # Khởi tạo tổ
        self.nests = [self.init_solution() for _ in range(self.num_nests)]
        self.fitness = [self.evaluate(nest) for nest in self.nests]

        best_idx = np.argmin(self.fitness)
        best_sol = self.nests[best_idx].copy()
        best_val = self.fitness[best_idx]

        for _ in range(self.max_iter):
            # Tạo tổ con bằng Lévy flight hoặc swap
            for i in range(self.num_nests):
                cuckoo = self.get_cuckoo(self.nests[i])
                cuckoo_fit = self.evaluate(cuckoo)

                # Lựa chọn tổ kém nhất ngẫu nhiên để thay thế nếu tốt hơn
                j = random.randint(0, self.num_nests - 1)
                if cuckoo_fit < self.fitness[j]:
                    self.nests[j] = cuckoo
                    self.fitness[j] = cuckoo_fit

                    if cuckoo_fit < best_val:
                        best_val = cuckoo_fit
                        best_sol = cuckoo.copy()

            # Một số tổ bị phát hiện (theo pa), tạo tổ mới
            for i in range(self.num_nests):
                if random.random() < self.pa:
                    self.nests[i] = self.init_solution()
                    self.fitness[i] = self.evaluate(self.nests[i])

            # Cập nhật best solution
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < best_val:
                best_val = self.fitness[current_best_idx]
                best_sol = self.nests[current_best_idx].copy()

        return best_sol, best_val
