import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        """
        distances: ma trận khoảng cách giữa các thành phố (2D numpy array)
        n_ants: số lượng kiến trong mỗi vòng lặp
        n_best: số kiến tốt nhất dùng để cập nhật pheromone
        n_iterations: số vòng lặp
        decay: hệ số bay hơi pheromone (rho)
        alpha: độ quan trọng của pheromone
        beta: độ quan trọng của heuristic (nghịch đảo khoảng cách)
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)  # pheromone khởi tạo
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
    
    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
                # print(f"Iteration {i+1}: new shortest path length = {shortest_path[1]}")
            # Bay hơi pheromone sau khi cập nhật
            self.pheromone *= (1 - self.decay)
        return all_time_shortest_path

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i]][path[i+1]]
        # Quay lại điểm bắt đầu (chu trình)
        total_dist += self.distances[path[-1]][path[0]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)  # Khởi đầu từ thành phố 0
            dist = self.gen_path_dist(path)
            all_paths.append((path, dist))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        path.append(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append(move)
            visited.add(move)
            prev = move
        return path

    def pick_move(self, pheromone, distances, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        distances = np.copy(distances)
        # Thay giá trị 0 khoảng cách bằng số rất nhỏ tránh chia 0
        distances[distances == 0] = 1e-10

        # Tính xác suất
        with np.errstate(divide='ignore', invalid='ignore'):
            row = (pheromone ** self.alpha) * ((1.0 / distances) ** self.beta)

        # Đặt xác suất = 0 cho thành phố đã thăm
        row[list(visited)] = 0

        total = np.sum(row)
        if total == 0 or np.isnan(total):
            # Nếu tổng xác suất là 0 hoặc NaN, tạo phân phối đều cho các thành phố chưa thăm
            row = np.zeros_like(row)
            for i in range(len(row)):
                if i not in visited:
                    row[i] = 1
            total = np.sum(row)

        norm_row = row / total

        # Lấy thành phố tiếp theo dựa trên phân phối xác suất
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move



    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in range(len(path) - 1):
                self.pheromone[path[move]][path[move+1]] += 1.0 / self.distances[path[move]][path[move+1]]
            # Cập nhật pheromone cho cạnh quay lại thành phố bắt đầu
            self.pheromone[path[-1]][path[0]] += 1.0 / self.distances[path[-1]][path[0]]
