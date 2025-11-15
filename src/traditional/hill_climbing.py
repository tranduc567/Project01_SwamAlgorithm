import numpy as np

def hill_climbing_continuous(func, dim, lower_bound, upper_bound, max_iter=500, step_size=0.1, verbose=False, seed=None):
    """
    Hill Climbing cho bài toán tối ưu hóa liên tục.

    Args:
        func (callable): Hàm mục tiêu nhận đầu vào vector numpy 1D, trả về giá trị fitness (càng nhỏ càng tốt).
        dim (int): Số chiều bài toán.
        lower_bound (float): Giới hạn dưới của mỗi chiều.
        upper_bound (float): Giới hạn trên của mỗi chiều.
        max_iter (int): Số vòng lặp tối đa.
        step_size (float): Bước nhảy để sinh điểm lân cận.
        verbose (bool): In tiến trình khi True.
        seed (int): Giá trị seed cho random (để tái lập kết quả).

    Returns:
        best_sol (np.ndarray): Điểm nghiệm tốt nhất tìm được.
        best_fit (float): Giá trị hàm mục tiêu tại best_sol.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Khởi tạo ngẫu nhiên điểm bắt đầu trong không gian tìm kiếm
    current_sol = np.random.uniform(lower_bound, upper_bound, dim)
    current_fit = func(current_sol)
    
    for iteration in range(max_iter):
        # Sinh điểm lân cận: cộng thêm vector ngẫu nhiên nhỏ trong [-step_size, step_size]
        neighbor = current_sol + np.random.uniform(-step_size, step_size, dim)
        # Giới hạn neighbor nằm trong biên
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        
        neighbor_fit = func(neighbor)
        
        # Nếu điểm lân cận tốt hơn thì cập nhật
        if neighbor_fit < current_fit:
            current_sol = neighbor
            current_fit = neighbor_fit
            if verbose:
                print(f"Iteration {iteration+1}: Improved fitness = {current_fit:.6f}")
    
    return current_sol, current_fit

def hill_climbing_tsp(dist_matrix, max_iter=1000):
    """
    Hill Climbing cho bài toán TSP.
    
    Args:
        dist_matrix (np.ndarray): Ma trận khoảng cách giữa các thành phố.
        max_iter (int): Số vòng lặp tối đa.
    
    Returns:
        best_path (list): Lộ trình tốt nhất tìm được.
        best_dist (float): Chi phí của lộ trình tốt nhất.
    """
    n = dist_matrix.shape[0]

    # Khởi tạo ngẫu nhiên một lời giải (chu trình)
    current_path = list(np.random.permutation(n))
    current_dist = path_distance(current_path, dist_matrix)

    for _ in range(max_iter):
        # Sinh một lời giải lân cận bằng cách hoán đổi 2 thành phố
        candidate_path = current_path.copy()
        i, j = np.random.choice(n, 2, replace=False)
        candidate_path[i], candidate_path[j] = candidate_path[j], candidate_path[i]

        candidate_dist = path_distance(candidate_path, dist_matrix)

        # Nếu lời giải mới tốt hơn, chấp nhận nó
        if candidate_dist < current_dist:
            current_path = candidate_path
            current_dist = candidate_dist

    return current_path, current_dist


def path_distance(path, dist_matrix):
    """Tính tổng khoảng cách cho chu trình đi qua các thành phố theo thứ tự trong path"""
    dist = 0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i]][path[i + 1]]
    dist += dist_matrix[path[-1]][path[0]]  # quay lại điểm đầu tiên
    return dist
