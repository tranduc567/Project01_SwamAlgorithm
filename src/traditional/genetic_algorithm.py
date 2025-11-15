import numpy as np

import numpy as np

def genetic_algorithm_continuous(func, dim, lower_bound, upper_bound, population_size=50, generations=500, mutation_rate=0.1, crossover_rate=0.8, verbose=False, seed=None):
    """
    Genetic Algorithm cho bài toán tối ưu hóa liên tục.

    Args:
        func (callable): Hàm mục tiêu nhận vector numpy 1D, trả về fitness (càng nhỏ càng tốt).
        dim (int): Số chiều bài toán.
        lower_bound (float): Giới hạn dưới của mỗi chiều.
        upper_bound (float): Giới hạn trên của mỗi chiều.
        population_size (int): Kích thước quần thể.
        generations (int): Số thế hệ.
        mutation_rate (float): Xác suất đột biến.
        crossover_rate (float): Xác suất lai ghép.
        verbose (bool): In tiến trình.
        seed (int): random seed.

    Returns:
        best_sol (np.ndarray): nghiệm tốt nhất tìm được.
        best_fit (float): giá trị hàm mục tiêu tại best_sol.
    """
    if seed is not None:
        np.random.seed(seed)

    # Khởi tạo quần thể ngẫu nhiên
    population = np.random.uniform(lower_bound, upper_bound, (population_size, dim))

    def fitness(ind):
        return func(ind)

    best_sol = None
    best_fit = np.inf

    for gen in range(generations):
        # Tính fitness cho toàn bộ quần thể
        fitnesses = np.array([fitness(ind) for ind in population])
        
        # Cập nhật nghiệm tốt nhất
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fit:
            best_fit = fitnesses[min_idx]
            best_sol = population[min_idx].copy()
            if verbose:
                print(f"Gen {gen+1}: Best fitness = {best_fit:.6f}")

        # Chọn lọc theo tỉ lệ nghịch fitness (tỉ lệ thuận với 1/fitness vì min)
        inv_fitness = 1.0 / (fitnesses + 1e-10)  # tránh chia 0
        probs = inv_fitness / np.sum(inv_fitness)

        new_population = []
        while len(new_population) < population_size:
            # Chọn cha mẹ
            parents_idx = np.random.choice(population_size, 2, replace=False, p=probs)
            parent1, parent2 = population[parents_idx[0]], population[parents_idx[1]]

            # Lai ghép (crossover)
            if np.random.rand() < crossover_rate:
                cross_point = np.random.randint(1, dim)
                child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Đột biến (mutation)
            for child in [child1, child2]:
                if np.random.rand() < mutation_rate:
                    mutate_idx = np.random.randint(dim)
                    child[mutate_idx] = np.random.uniform(lower_bound, upper_bound)

            new_population.extend([child1, child2])

        population = np.array(new_population[:population_size])

    return best_sol, best_fit


def genetic_algorithm_tsp(dist_matrix, population_size=50, generations=200, mutation_rate=0.1):
    """
    Giải thuật di truyền (Genetic Algorithm) cho bài toán TSP.

    Args:
        dist_matrix (np.ndarray): Ma trận khoảng cách.
        population_size (int): Kích thước quần thể.
        generations (int): Số thế hệ tối đa.
        mutation_rate (float): Xác suất đột biến.

    Returns:
        best_path (list): Lộ trình tốt nhất tìm được.
        best_dist (float): Chi phí lộ trình tốt nhất.
    """
    n = dist_matrix.shape[0]

    # Khởi tạo quần thể ngẫu nhiên (danh sách các cá thể - mỗi cá thể là 1 permuation)
    population = [list(np.random.permutation(n)) for _ in range(population_size)]

    def fitness(path):
        return 1.0 / path_distance(path, dist_matrix)

    def selection(pop, fitnesses):
        # Chọn 2 cá thể theo roulette wheel selection dựa trên fitness
        probs = fitnesses / np.sum(fitnesses)
        selected = np.random.choice(len(pop), size=2, replace=False, p=probs)
        return pop[selected[0]], pop[selected[1]]

    def crossover(parent1, parent2):
        # Order crossover (OX)
        start, end = sorted(np.random.choice(range(n), 2, replace=False))
        child = [None]*n
        # Copy đoạn con giữa từ parent1
        child[start:end+1] = parent1[start:end+1]

        # Điền các thành phố còn lại theo thứ tự parent2
        p2_idx = 0
        for i in range(n):
            if child[i] is None:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
        return child

    def mutate(path):
        # Swap mutation
        if np.random.rand() < mutation_rate:
            i, j = np.random.choice(n, 2, replace=False)
            path[i], path[j] = path[j], path[i]
        return path

    best_path = None
    best_dist = np.inf

    for _ in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        new_population = []

        # Giữ lại cá thể tốt nhất
        idx_best = np.argmax(fitnesses)
        if 1.0/fitnesses[idx_best] < best_dist:
            best_dist = 1.0/fitnesses[idx_best]
            best_path = population[idx_best].copy()
        new_population.append(best_path.copy())

        # Tạo thế hệ mới
        while len(new_population) < population_size:
            parent1, parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return best_path, best_dist


def path_distance(path, dist_matrix):
    dist = 0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i]][path[i + 1]]
    dist += dist_matrix[path[-1]][path[0]]
    return dist
