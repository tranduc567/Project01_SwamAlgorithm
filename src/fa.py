import numpy as np


class FA:
    def __init__(self, obj_func, lb, ub, n_dims, pop_size=10, epochs=100,
                 max_sparks=50, p_a=0.04, p_b=0.8, max_ea=40, m_sparks=50, seed=None):

        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.n_dims = n_dims

        self.pop_size = pop_size
        self.epochs = epochs

        self.max_sparks = max_sparks
        self.p_a = p_a
        self.p_b = p_b
        self.max_ea = max_ea
        self.m_sparks = m_sparks

        self.rng = np.random.default_rng(seed)

        # ---- Initialize population ----
        self.pop = self.rng.uniform(self.lb, self.ub, (pop_size, n_dims))
        self.fitness = np.apply_along_axis(self.obj_func, 1, self.pop)

        # ---- Best ----
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.pop[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]

    # Clip to bound
    def correct_solution(self, sol):
        return np.clip(sol, self.lb, self.ub)

    # -----------------------------------------------------
    #  Main FA Algorithm
    # -----------------------------------------------------
    def solve(self):
        history = []

        for ep in range(1, self.epochs + 1):

            fit_sorted = np.sort(self.fitness)
            best_f = fit_sorted[0]
            worst_f = fit_sorted[-1]
            sum_f = np.sum(fit_sorted)

            sparks_population = []

            # =========================
            # 1. Explosion for each firework
            # =========================
            for i in range(self.pop_size):
                f_i = self.fitness[i]

                # Number of sparks si
                si = self.max_sparks * (worst_f - f_i) / (self.pop_size * worst_f - sum_f + 1e-12)

                # Explosion amplitude Ai
                Ai = self.max_ea * (f_i - best_f) / (sum_f - best_f + 1e-12)

                # Constraint si
                if si < self.p_a * self.max_sparks:
                    si = int(self.p_a * self.max_sparks) + 1
                elif si > self.p_b * self.max_sparks:
                    si = int(self.p_b * self.max_sparks) + 1
                else:
                    si = int(si) + 1

                # ==== Generate sparks ====
                for _ in range(si):
                    pos_new = self.pop[i].copy()

                    # Random subset of dimensions
                    num_dims = int(self.rng.uniform() * self.n_dims)
                    if num_dims == 0:
                        continue
                    idxs = self.rng.choice(range(self.n_dims), num_dims, replace=False)

                    displacement = Ai * self.rng.uniform(-1, 1)

                    pos_new[idxs] = pos_new[idxs] + displacement

                    pos_new = self.correct_solution(pos_new)
                    sparks_population.append(pos_new)

            # =========================
            # 2. Gaussian Sparks
            # =========================
            for _ in range(self.m_sparks):
                idx = self.rng.integers(0, self.pop_size)
                pos_new = self.pop[idx].copy()

                num_dims = int(self.rng.uniform() * self.n_dims)
                if num_dims == 0:
                    continue

                idxs = self.rng.choice(range(self.n_dims), num_dims, replace=False)
                pos_new[idxs] += self.rng.normal(0, 1, num_dims)

                pos_new = self.correct_solution(pos_new)
                sparks_population.append(pos_new)

            # =========================
            # 3. Evaluate all sparks
            # =========================
            sparks_population = np.array(sparks_population)
            sparks_fitness = np.apply_along_axis(self.obj_func, 1, sparks_population)

            # Merge population
            merged_pop = np.vstack([self.pop, sparks_population])
            merged_fit = np.hstack([self.fitness, sparks_fitness])

            # Keep the best pop_size
            idx_sorted = np.argsort(merged_fit)
            self.pop = merged_pop[idx_sorted[:self.pop_size]]
            self.fitness = merged_fit[idx_sorted[:self.pop_size]]

            # =========================
            # 4. Update global best
            # =========================
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.pop[best_idx].copy()

            history.append([self.best_solution.copy(), self.best_fitness])

            # print(f"Epoch {ep}: Best = {self.best_fitness}")

        return self.best_solution, self.best_fitness, history