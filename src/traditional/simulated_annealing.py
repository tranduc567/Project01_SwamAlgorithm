import numpy as np

import numpy as np

def simulated_annealing_continuous(objective_func, dim, lower_bound, upper_bound,
                                   initial_temp=1000, cooling_rate=0.995, max_iter=500, step_size=0.1):
    """
    Simulated Annealing cho bài toán tối ưu hóa liên tục.

    Args:
        objective_func (callable): Hàm mục tiêu f(x), x là vector.
        dim (int): Số chiều của bài toán.
        lower_bound (float): Giới hạn dưới của từng chiều.
        upper_bound (float): Giới hạn trên của từng chiều.
        initial_temp (float): Nhiệt độ ban đầu.
        cooling_rate (float): Tốc độ làm lạnh (0 < cooling_rate < 1).
        max_iter (int): Số vòng lặp tối đa.
        step_size (float): Kích thước bước tạo lời giải lân cận (độ lớn nhiễu Gaussian).

    Returns:
        best_solution (np.ndarray): Nghiệm tốt nhất tìm được.
        best_fitness (float): Giá trị hàm mục tiêu tốt nhất.
    """
    # Khởi tạo nghiệm ban đầu ngẫu nhiên trong khoảng cho trước
    current_solution = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    current_fitness = objective_func(current_solution)

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    temp = initial_temp

    for _ in range(max_iter):
        # Tạo lời giải lân cận bằng cách thêm nhiễu Gaussian
        candidate_solution = current_solution + np.random.normal(0, step_size, size=dim)
        # Giới hạn candidate trong biên
        candidate_solution = np.clip(candidate_solution, lower_bound, upper_bound)
        candidate_fitness = objective_func(candidate_solution)

        delta = candidate_fitness - current_fitness  # Giả sử bài toán minimization

        # Chấp nhận lời giải mới nếu tốt hơn hoặc với xác suất theo nhiệt độ
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_solution = candidate_solution
            current_fitness = candidate_fitness

            # Cập nhật nghiệm tốt nhất
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

        # Giảm nhiệt độ
        temp *= cooling_rate
        if temp < 1e-10:
            break

    return best_solution, best_fitness

def simulated_annealing_tsp(dist_matrix, initial_temp=1000, cooling_rate=0.995, max_iter=1000):
    """
    Giải thuật Simulated Annealing cho bài toán TSP.

    Args:
        dist_matrix (np.ndarray): Ma trận khoảng cách.
        initial_temp (float): Nhiệt độ ban đầu.
        cooling_rate (float): Tốc độ làm lạnh (0 < cooling_rate < 1).
        max_iter (int): Số vòng lặp tối đa.

    Returns:
        best_path (list): Lộ trình tốt nhất tìm được.
        best_dist (float): Chi phí lộ trình tốt nhất.
    """
    n = dist_matrix.shape[0]
    
    # Khởi tạo lời giải ban đầu ngẫu nhiên
    current_path = list(np.random.permutation(n))
    current_dist = path_distance(current_path, dist_matrix)
    
    best_path = current_path.copy()
    best_dist = current_dist

    temp = initial_temp

    for _ in range(max_iter):
        # Tạo lời giải lân cận bằng cách hoán đổi 2 thành phố
        candidate_path = current_path.copy()
        i, j = np.random.choice(n, 2, replace=False)
        candidate_path[i], candidate_path[j] = candidate_path[j], candidate_path[i]
        candidate_dist = path_distance(candidate_path, dist_matrix)

        # Chấp nhận lời giải mới nếu tốt hơn hoặc với xác suất dựa trên nhiệt độ
        delta = candidate_dist - current_dist
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_path = candidate_path
            current_dist = candidate_dist

            # Cập nhật lời giải tốt nhất nếu cần
            if current_dist < best_dist:
                best_path = current_path.copy()
                best_dist = current_dist

        # Giảm nhiệt độ
        temp *= cooling_rate

        if temp < 1e-10:
            break

    return best_path, best_dist


def path_distance(path, dist_matrix):
    """Tính tổng khoảng cách cho chu trình đi qua các thành phố theo thứ tự trong path"""
    dist = 0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i]][path[i + 1]]
    dist += dist_matrix[path[-1]][path[0]]  # quay lại điểm đầu tiên
    return dist
