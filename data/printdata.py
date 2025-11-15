import numpy as np

# Load file distances.npy
distances = np.load("data/distances.npy")

# In ra kích thước ma trận
print("Shape of distances:", distances.shape)

# In ra toàn bộ ma trận khoảng cách (cẩn thận nếu ma trận lớn!)
print("Distance matrix:")
print(distances)

# Hoặc in một phần (ví dụ 5 dòng đầu)
print("First 5 rows:")
print(distances[:5])
