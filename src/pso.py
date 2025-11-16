import numpy as np


class PSO:
    def __init__(self, obj_func, lb, ub, n_dims, pop_size=30, epochs=100,
                 c1=2.05, c2=2.05, w=0.4, seed=None):

        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = n_dims
        self.pop_size = pop_size
        self.epochs = epochs

        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.rng = np.random.default_rng(seed)

        # ---- Khởi tạo quần thể ----
        self.pop = self.rng.uniform(self.lb, self.ub, (pop_size, n_dims))
        self.vel = self.rng.uniform(-(self.ub - self.lb), (self.ub - self.lb), (pop_size, n_dims))

        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # Best cục bộ
        self.local_best_pos = self.pop.copy()
        self.local_best_fit = self.fitness.copy()

        # Best toàn cục
        g_idx = np.argmin(self.fitness)
        self.global_best_pos = self.pop[g_idx].copy()
        self.global_best_fit = self.fitness[g_idx]

    def correct_solution(self, sol):
        return np.clip(sol, self.lb, self.ub)

    def solve(self):
        history = []

        for ep in range(1, self.epochs + 1):

            for i in range(self.pop_size):

                r1 = self.rng.random(self.n_dims)
                r2 = self.rng.random(self.n_dims)

                cognitive = self.c1 * r1 * (self.local_best_pos[i] - self.pop[i])
                social = self.c2 * r2 * (self.global_best_pos - self.pop[i])

                # ---- Update velocity ----
                self.vel[i] = self.w * self.vel[i] + cognitive + social

                # ---- Update position ----
                new_pos = self.pop[i] + self.vel[i]
                new_pos = self.correct_solution(new_pos)
                new_fit = self.obj_func(new_pos)

                # Cập nhật individual best
                if new_fit < self.local_best_fit[i]:
                    self.local_best_pos[i] = new_pos.copy()
                    self.local_best_fit[i] = new_fit

                # Cập nhật global best
                if new_fit < self.global_best_fit:
                    self.global_best_pos = new_pos.copy()
                    self.global_best_fit = new_fit

                # Update current particle
                self.pop[i] = new_pos.copy()
                self.fitness[i] = new_fit

            # Lưu lịch sử
            history.append([self.global_best_pos.copy(), self.global_best_fit])

        return self.global_best_pos, self.global_best_fit, history, 