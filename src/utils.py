import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Dùng để vẽ 3Dplot


def ackley_function(x):
    """
    Hàm Ackley, đầu vào x là numpy array 1D.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.exp(1)

def ackley_function_2d(x, y):
    """
    Hàm Ackley 2D, phục vụ cho vẽ surface plot.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = x**2 + y**2
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return term1 + term2 + a + np.exp(1)

def plot_ackley_surface(lower_bound=-32.768, upper_bound=32.768, step=0.5):
    """
    Vẽ surface plot 3D của hàm Ackley 2D.
    """
    x = np.arange(lower_bound, upper_bound, step)
    y = np.arange(lower_bound, upper_bound, step)
    X, Y = np.meshgrid(x, y)
    Z = ackley_function_2d(X, Y)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Ackley Function Landscape')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
