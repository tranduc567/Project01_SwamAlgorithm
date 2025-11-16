import numpy as np

def sa_continuous(obj_func, n_dims, lb, ub, epochs, T_init=100, alpha=0.99):
    current = np.random.uniform(lb, ub, n_dims)
    best = current.copy()
    best_val = obj_func(best)
    current_val = best_val
    T = T_init
    history = []

    for _ in range(epochs):
        neighbor = current + np.random.uniform(-0.1,0.1,n_dims)
        neighbor = np.clip(neighbor, lb, ub)
        val = obj_func(neighbor)
        delta = val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta/T):
            current = neighbor
            current_val = val
            if val < best_val:
                best = neighbor
                best_val = val
        history.append([best, best_val])
        T *= alpha
    return best, best_val, history



