import numpy as np

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
