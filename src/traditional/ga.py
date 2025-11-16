import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================
# Genetic Algorithm cho TSP
# ==========================
def ga_tsp(distance_matrix, pop_size=100, epoch=500, mutation_rate=0.2, seed=None):
    """
    GA cho TSP: representation là list các thành phố (hoán vị)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_cities = distance_matrix.shape[0]

    # ----------- Khởi tạo quần thể ----------- 
    def create_individual():
        ind = list(range(n_cities))
        random.shuffle(ind)
        return ind

    population = [create_individual() for _ in range(pop_size)]

    # ----------- Hàm fitness ----------- 
    def fitness(ind):
        dist = sum(distance_matrix[ind[i-1], ind[i]] for i in range(n_cities))
        return dist

    # ----------- Selection: Tournament ----------- 
    def select_parent(pop, k=3):
        candidates = random.sample(pop, k)
        candidates.sort(key=fitness)
        return candidates[0]

    # ----------- Crossover: Order Crossover (OX) ----------- 
    def crossover(parent1, parent2):
        a, b = sorted(random.sample(range(n_cities), 2))
        child = [None]*n_cities
        child[a:b] = parent1[a:b]
        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = city
        return child

    # ----------- Mutation: swap mutation ----------- 
    def mutate(ind):
        if random.random() < mutation_rate:
            a, b = random.sample(range(n_cities), 2)
            ind[a], ind[b] = ind[b], ind[a]

    # ----------- Evolution ----------- 
    best_individual = None
    best_distance = float('inf')
    history = []

    for _ in range(epoch):
        new_population = []
        for _ in range(pop_size):
            p1 = select_parent(population)
            p2 = select_parent(population)
            child = crossover(p1, p2)
            mutate(child)
            new_population.append(child)

        population = new_population

        # Update best
        for ind in population:
            dist = fitness(ind)
            if dist < best_distance:
                best_distance = dist
                best_individual = ind.copy()

        history.append([best_individual, best_distance])

    return best_individual, best_distance, history
