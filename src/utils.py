import numpy as np

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
