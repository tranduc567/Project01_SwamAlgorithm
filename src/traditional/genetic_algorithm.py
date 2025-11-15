import numpy as np

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
