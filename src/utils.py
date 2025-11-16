def get_converge_epoch(history, best_fit):
    """
    Trả về epoch nhỏ nhất mà đạt best_fit dựa trên history.

    Args:
        history: danh sách [best_pos, best_fit] qua các epoch
        best_fit: giá trị fitness tốt nhất tìm được

    Returns:
        epoch (int) nhỏ nhất đạt best_fit, hoặc None nếu không tìm thấy
    """
    for idx, (_, fit)in enumerate(history, start=1):  # start=1 để epoch bắt đầu từ 1
        if fit == best_fit:
            return idx
    return None
 