import numpy as np
import random

class ParticleSwarmOptimization:
    def __init__(self,
                 num_particles=20,
                 max_iter=50,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 mode="continuous",
                 lower_bound=-5,
                 upper_bound=5,
                 dim=10,
                 distance_matrix=None):

        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mode = mode
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.distance_matrix = distance_matrix

        self.positions = []
        self.velocities = []

        self.pbest_positions = []
        self.pbest_fitness = []

        self.gbest_position = None
        self.gbest_fitness = float('inf')

    def tsp_distance(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.distance_matrix[path[i], path[i + 1]]
        d += self.distance_matrix[path[-1], path[0]]
        return d

    def continuous_fitness(self, x):
        return np.sum(x ** 2)

    def evaluate(self, solution):
        if self.mode == "discrete":
            return self.tsp_distance(solution)
        else:
            return self.continuous_fitness(solution)

    def init_position(self):
        if self.mode == "discrete":
            return np.random.permutation(len(self.distance_matrix))
        else:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def init_velocity(self):
        if self.mode == "discrete":
            return None
        else:
            return np.random.uniform(-(self.upper_bound - self.lower_bound),
                                     self.upper_bound - self.lower_bound,
                                     self.dim)

    def velocity_update(self, vi, xi, pi, g):
        r1, r2 = np.random.rand(), np.random.rand()
        return self.w * vi + self.c1 * r1 * (pi - xi) + self.c2 * r2 * (g - xi)

    def position_update_continuous(self, xi, vi):
        x_new = xi + vi
        x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
        return x_new

    def apply_swap(self, current, target):
        # swap 1 hoặc 2 lần để làm current gần target hơn
        current = current.copy()
        diff_positions = [i for i in range(len(current)) if current[i] != target[i]]
        n_swaps = min(2, len(diff_positions) // 2)  # swap tối đa 2 lần
        for _ in range(n_swaps):
            if len(diff_positions) < 2:
                break
            i, j = random.sample(diff_positions, 2)
            current[i], current[j] = current[j], current[i]
            # cập nhật lại danh sách vị trí khác nhau
            diff_positions = [idx for idx in range(len(current)) if current[idx] != target[idx]]
        return current

    def position_update_discrete(self, xi, pi, g):
        # Chọn cập nhật dựa trên pbest hoặc gbest
        if random.random() < 0.5:
            return self.apply_swap(xi, pi)
        else:
            return self.apply_swap(xi, g)

    def run(self):
        self.positions = [self.init_position() for _ in range(self.num_particles)]
        if self.mode == "continuous":
            self.velocities = [self.init_velocity() for _ in range(self.num_particles)]
        else:
            self.velocities = [None for _ in range(self.num_particles)]

        self.pbest_positions = [pos.copy() for pos in self.positions]
        self.pbest_fitness = [self.evaluate(pos) for pos in self.positions]

        best_idx = np.argmin(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]

        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                if self.mode == "continuous":
                    self.velocities[i] = self.velocity_update(self.velocities[i],
                                                              self.positions[i],
                                                              self.pbest_positions[i],
                                                              self.gbest_position)
                    self.positions[i] = self.position_update_continuous(self.positions[i],
                                                                        self.velocities[i])
                else:
                    self.positions[i] = self.position_update_discrete(self.positions[i],
                                                                      self.pbest_positions[i],
                                                                      self.gbest_position)

                fitness_new = self.evaluate(self.positions[i])

                if fitness_new < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness_new
                    self.pbest_positions[i] = self.positions[i].copy()

                    if fitness_new < self.gbest_fitness:
                        self.gbest_fitness = fitness_new
                        self.gbest_position = self.positions[i].copy()

        return self.gbest_position, self.gbest_fitness


import numpy as np
import random

class ParticleSwarmOptimizationTSP:
    def __init__(self,
                 distance_matrix,
                 num_particles=30,
                 max_iter=100,
                 w=0.8,
                 c1=1.5,
                 c2=1.5):
        self.distance_matrix = distance_matrix
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = len(distance_matrix)

        self.positions = []  # mỗi vị trí là 1 permutation
        self.velocities = []  # mỗi velocity là list các swap thao tác [(i,j), ...]

        self.pbest_positions = []
        self.pbest_fitness = []

        self.gbest_position = None
        self.gbest_fitness = float('inf')

    def tsp_distance(self, path):
        d = 0
        for i in range(len(path) - 1):
            d += self.distance_matrix[path[i], path[i + 1]]
        d += self.distance_matrix[path[-1], path[0]]
        return d

    def init_position(self):
        return np.random.permutation(self.dim)

    def get_swap_sequence(self, current, target):
        # Tạo chuỗi các swap để chuyển current thành target
        current = current.copy()
        swaps = []
        pos_in_current = {v: i for i, v in enumerate(current)}
        for i in range(len(current)):
            if current[i] != target[i]:
                swap_with_index = pos_in_current[target[i]]
                # swap i và swap_with_index
                swaps.append((i, swap_with_index))
                # thực hiện swap trong current
                val_i, val_j = current[i], current[swap_with_index]
                current[i], current[swap_with_index] = val_j, val_i
                # cập nhật vị trí sau swap
                pos_in_current[val_i] = swap_with_index
                pos_in_current[val_j] = i
        return swaps

    def apply_swaps(self, position, swaps):
        # Áp dụng chuỗi swap lên vị trí
        position = position.copy()
        for i, j in swaps:
            position[i], position[j] = position[j], position[i]
        return position

    def multiply_swaps(self, swaps, factor):
        # Giữ lại chỉ factor phần swaps (lấy phần đầu)
        k = int(len(swaps) * factor)
        return swaps[:k]

    def add_swaps(self, swaps1, swaps2):
        # Nối 2 chuỗi swap
        return swaps1 + swaps2

    def velocity_update(self, velocity, current_pos, pbest_pos, gbest_pos):
        # velocity cũ * w
        vel_inertia = self.multiply_swaps(velocity, self.w) if velocity is not None else []

        # swaps từ current -> pbest
        swaps_pbest = self.get_swap_sequence(current_pos, pbest_pos)
        swaps_pbest = self.multiply_swaps(swaps_pbest, self.c1 * random.random())

        # swaps từ current -> gbest
        swaps_gbest = self.get_swap_sequence(current_pos, gbest_pos)
        swaps_gbest = self.multiply_swaps(swaps_gbest, self.c2 * random.random())

        # cộng tất cả lại
        new_velocity = self.add_swaps(vel_inertia, swaps_pbest)
        new_velocity = self.add_swaps(new_velocity, swaps_gbest)

        # Có thể giảm độ dài velocity nếu quá dài (để tránh quá nhiều swap)
        max_velocity_length = self.dim * 2
        if len(new_velocity) > max_velocity_length:
            new_velocity = new_velocity[:max_velocity_length]

        return new_velocity

    def run(self):
        self.positions = [self.init_position() for _ in range(self.num_particles)]
        self.velocities = [[] for _ in range(self.num_particles)]  # velocity là list các swap thao tác

        self.pbest_positions = [pos.copy() for pos in self.positions]
        self.pbest_fitness = [self.tsp_distance(pos) for pos in self.positions]

        best_idx = np.argmin(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_fitness = self.pbest_fitness[best_idx]

        for iter in range(self.max_iter):
            for i in range(self.num_particles):
                # cập nhật velocity
                self.velocities[i] = self.velocity_update(self.velocities[i],
                                                          self.positions[i],
                                                          self.pbest_positions[i],
                                                          self.gbest_position)
                # cập nhật vị trí mới bằng cách áp dụng velocity
                self.positions[i] = self.apply_swaps(self.positions[i], self.velocities[i])

                # tính fitness mới
                fitness_new = self.tsp_distance(self.positions[i])

                # cập nhật cá nhân tốt nhất
                if fitness_new < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness_new
                    self.pbest_positions[i] = self.positions[i].copy()

                    # cập nhật toàn cục tốt nhất
                    if fitness_new < self.gbest_fitness:
                        self.gbest_fitness = fitness_new
                        self.gbest_position = self.positions[i].copy()

            print(f"Iter {iter+1}/{self.max_iter} - Best distance: {self.gbest_fitness:.4f}")

        return self.gbest_position, self.gbest_fitness
