import numpy as np


class CSA:
    def __init__(self, obj_func, lb, ub, n_dims, pop_size=30, epochs=100, p_a=0.3, seed=None):

        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = n_dims
        self.pop_size = pop_size
        self.epochs = epochs
        self.p_a = p_a

        self.n_abandon = int(p_a * pop_size)
        self.rng = np.random.default_rng(seed)

        # ---- Khởi tạo tổ ----
        self.pop = self.rng.uniform(self.lb, self.ub, (pop_size, n_dims))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # Best global
        idx = np.argmin(self.fitness)
        self.best_pos = self.pop[idx].copy()
        self.best_fit = self.fitness[idx]

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def levy_flight(self, beta=1.5):
        """
        Levy flight theo công thức Mantegna
        """
        sigma_u = (self.rng.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (self.rng.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2))) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, self.n_dims)
        v = self.rng.normal(0, 1, self.n_dims)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def solve(self):
        history = []

        for ep in range(1, self.epochs + 1):

            for i in range(self.pop_size):
                # ---- Levy flight ----
                step = self.levy_flight()
                new_pos = self.pop[i] + 0.01 * step * (self.pop[i] - self.best_pos)

                new_pos = self.correct(new_pos)
                new_fit = self.obj_func(new_pos)

                # Cập nhật nếu tốt hơn
                if new_fit < self.fitness[i]:
                    self.pop[i] = new_pos
                    self.fitness[i] = new_fit

            # ---- Abandon worst nests ----
            worst_idx = np.argsort(self.fitness)[-self.n_abandon:]  # tổ tệ nhất

            for i in worst_idx:
                self.pop[i] = self.rng.uniform(self.lb, self.ub, self.n_dims)
                self.fitness[i] = self.obj_func(self.pop[i])

            # ---- Cập nhật best global ----
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fit:
                self.best_pos = self.pop[best_idx].copy()
                self.best_fit = self.fitness[best_idx]

            history.append([self.best_pos.copy(), self.best_fit])

        return self.best_pos, self.best_fit, history