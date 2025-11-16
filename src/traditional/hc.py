import numpy as np
import time
from utils import get_converge_epoch
import matplotlib.pyplot as plt
from ackley import ackley


def hill_climbing_continuous(obj_func, n_dims, lb, ub, epochs=1000):
    """Liên tục: tối ưu hàm số"""
    current = np.random.uniform(lb, ub, n_dims)
    best = current.copy()
    best_val = obj_func(best)
    history = []

    for _ in range(epochs):
        neighbor = best + np.random.uniform(-0.1, 0.1, n_dims)
        neighbor = np.clip(neighbor, lb, ub)
        val = obj_func(neighbor)
        if val < best_val:
            best = neighbor
            best_val = val
        history.append([best, best_val])
    return best, best_val, history


