import numpy as np
import random

class ABC:
    def __init__(self, 
                 num_bees=30,
                 limit=50,
                 max_iter=200,
                 mode="discrete",   # "discrete" for TSP, "continuous" for benchmark
                 lower_bound=-5,
                 upper_bound=5,
                 dim=10,
                 distance_matrix=None):

        self.num_bees = num_bees
        self.limit = limit
        self.max_iter = max_iter
        self.mode = mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim

        # TSP dùng matrix
        self.distance_matrix = distance_matrix

        # Lưu trữ
        self.food_sources = []
        self.fitness = []
        self.trial = []

    # --------------------------
    #  FITNESS
    # --------------------------
    def tsp_distance(self, path):
        """Tính tổng độ dài tour"""
        d = 0
        for i in range(len(path)-1):
            d += self.distance_matrix[path[i], path[i+1]]
        d += self.distance_matrix[path[-1], path[0]]
        return d

    def continuous_f(self, x):
        """Ví dụ Sphere"""
        return np.sum(x**2)

    def evaluate(self, solution):
        if self.mode == "discrete":
            return self.tsp_distance(solution)
        else:
            return self.continuous_f(solution)

    # --------------------------
    #  KHỞI TẠO
    # --------------------------
    def init_solution(self):
        if self.mode == "discrete":
            return np.random.permutation(len(self.distance_matrix))
        else:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def neighbor(self, solution):
        if self.mode == "discrete":
            # swap 2 vị trí (TSP)
            i, j = random.sample(range(len(solution)), 2)
            new = solution.copy()
            new[i], new[j] = new[j], new[i]
            return new
        else:
            # perturb continuous
            phi = np.random.uniform(-1, 1, self.dim)
            return solution + phi * (solution - np.random.uniform(self.lower_bound, self.upper_bound, self.dim))

    # --------------------------
    #  MAIN RUN
    # --------------------------
    def run(self):
        # init
        self.food_sources = [self.init_solution() for _ in range(self.num_bees)]
        self.fitness = [self.evaluate(sol) for sol in self.food_sources]
        self.trial = [0] * self.num_bees

        best_sol = self.food_sources[np.argmin(self.fitness)].copy()
        best_val = np.min(self.fitness)

        for _ in range(self.max_iter):

            # 1. Employed Bees
            for i in range(self.num_bees):
                new_sol = self.neighbor(self.food_sources[i])
                new_fit = self.evaluate(new_sol)
                if new_fit < self.fitness[i]:
                    self.food_sources[i] = new_sol
                    self.fitness[i] = new_fit
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # 2. Onlooker Bees
            prob = (1 / (1 + np.array(self.fitness)))
            prob /= np.sum(prob)

            for _ in range(self.num_bees):
                i = np.random.choice(range(self.num_bees), p=prob)
                new_sol = self.neighbor(self.food_sources[i])
                new_fit = self.evaluate(new_sol)
                if new_fit < self.fitness[i]:
                    self.food_sources[i] = new_sol
                    self.fitness[i] = new_fit
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1

            # 3. Scout Bees
            for i in range(self.num_bees):
                if self.trial[i] > self.limit:
                    self.food_sources[i] = self.init_solution()
                    self.fitness[i] = self.evaluate(self.food_sources[i])
                    self.trial[i] = 0

            # update best
            if np.min(self.fitness) < best_val:
                idx = np.argmin(self.fitness)
                best_val = self.fitness[idx]
                best_sol = self.food_sources[idx].copy()

        return best_sol, best_val
