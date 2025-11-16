import numpy as np
import random
from typing import List, Callable


# =========================
# Generic ACO Problem
# =========================
class ACOProblem:
    """Generic ACO problem: mỗi problem định nghĩa cách xây dựng solution."""
    def __init__(self, n_decisions: int, objective: Callable):
        self.n_decisions = n_decisions
        self.objective = objective
        self.solution = None
        self.current_index = 0

    def feasible_solution(self) -> List[int]:
        raise NotImplementedError

    def make_decision(self, index: int, value: int):
        raise NotImplementedError

    def is_complete(self) -> bool:
        raise NotImplementedError

    def build(self) -> List[int]:
        raise NotImplementedError

    def evaluate(self, solution: List[int]) -> float:
        return self.objective(solution)

    def reset(self):
        self.solution = [None] * self.n_decisions
        self.current_index = 0


# =========================
# General ACO Optimizer
# =========================
class ACO:
    def __init__(self, problem: ACOProblem, n_ants=10, epochs=100,
                 alpha=1.0, beta=2.0, rho=0.5, eta=None):
        self.problem = problem
        self.n_ants = n_ants
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        # ensure eta is finite and same shape as pheromone expectation
        self.eta = eta if eta is not None else np.ones((problem.n_decisions, problem.n_decisions))
        # replace inf/nan in eta
        self.eta = np.nan_to_num(self.eta, nan=1.0, posinf=1.0, neginf=1.0)
        self.pheromone = np.ones((problem.n_decisions, problem.n_decisions))
        self.history = []

    def _construct_ant_solution(self) -> List[int]:
        """
        Robust construction:
        - if feasible is empty: break early
        - compute tau, eta; clamp to small positive min
        - compute probs, clip negatives, if sum==0 -> uniform fallback
        """
        self.problem.reset()
        while not self.problem.is_complete():
            feasible = self.problem.feasible_solution()
            if not feasible:
                break  # no feasible move: finish early

            # make sure feasible is list of ints
            feasible = list(feasible)

            # build tau and eta arrays for feasible choices
            # use row = current_index as default (works for TSP/GraphColoring/Knapsack as implemented)
            row = min(self.problem.current_index, self.problem.n_decisions - 1)
            tau = np.array([self.pheromone[row, i] for i in feasible], dtype=float) ** self.alpha
            eta = np.array([self.eta[row, i] for i in feasible], dtype=float) ** self.beta

            # sanitize values: replace nan/inf with small positive, clip negative
            tau = np.nan_to_num(tau, nan=1e-12, posinf=1e-12, neginf=1e-12)
            eta = np.nan_to_num(eta, nan=1e-12, posinf=1e-12, neginf=1e-12)
            tau = np.clip(tau, 1e-12, None)
            eta = np.clip(eta, 1e-12, None)

            probs = tau * eta

            # clip accidental tiny negatives, then check sum
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s <= 0 or not np.isfinite(s):
                # fallback: uniform random among feasible
                choice = random.choice(feasible)
            else:
                probs = probs / s
                # final safety: if any probs are negative or nan, fallback to uniform
                if np.any(np.isnan(probs)) or np.any(probs < 0):
                    choice = random.choice(feasible)
                else:
                    choice = np.random.choice(feasible, p=probs)

            self.problem.make_decision(self.problem.current_index, choice)

        return self.problem.build()

    def _update_pheromone(self, solutions: List[List[int]]):
        # evaporation
        self.pheromone *= (1 - self.rho)

        # deposit pheromone: be robust when val can be zero or inf
        for sol in solutions:
            val = self.problem.evaluate(sol)
            # if objective is zero or negative (e.g., negative cost for knapsack), we protect denom
            denom = val if (val > 0) else (abs(val) + 1e-10)
            for i, x in enumerate(sol):
                # index row: use i but clamp just in case
                row = min(i, self.problem.n_decisions - 1)
                col = int(x)
                # ensure indices in range
                if 0 <= row < self.pheromone.shape[0] and 0 <= col < self.pheromone.shape[1]:
                    self.pheromone[row, col] += 1.0 / (denom + 1e-10)

    def solve(self):
        best_solution = None
        best_value = float('inf')
        for epoch in range(self.epochs):
            solutions = [self._construct_ant_solution() for _ in range(self.n_ants)]
            self._update_pheromone(solutions)
            for sol in solutions:
                val = self.problem.evaluate(sol)
                if val < best_value:
                    best_value = val
                    best_solution = sol
            self.history.append([best_solution, best_value])
        return best_solution, best_value, self.history