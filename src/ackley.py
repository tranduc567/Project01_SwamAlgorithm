import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -------------------------
#  Ackley function
# -------------------------
def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.array(x)
    d = x.size

    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    return term1 + term2 + a + np.e


# -------------------------
#  Visualization
# -------------------------

# Grid for x,y
# n = 200
# x = np.linspace(-5, 5, n)
# y = np.linspace(-5, 5, n)
# X, Y = np.meshgrid(x, y)

# # Compute Ackley on grid
# Z = np.zeros_like(X)
# for i in range(n):
#     for j in range(n):
#         Z[i, j] = ackley([X[i, j], Y[i, j]])


# # ---------------------------------------------------------
# # 1) 3D SURFACE
# # ---------------------------------------------------------
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=0.3)

# ax.set_title("Ackley function — 3D surface")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("f(x,y)")
# plt.show()


# # ---------------------------------------------------------
# # 2) 3D WIREFRAME
# # ---------------------------------------------------------
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, linewidth=0.35)

# ax.set_title("Ackley function — 3D wireframe")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("f(x,y)")
# plt.show()


# # ---------------------------------------------------------
# # 3) 2D CONTOUR
# # ---------------------------------------------------------
# plt.figure(figsize=(10, 8))

# cp = plt.contourf(X, Y, Z, levels=50)
# plt.colorbar(cp)

# plt.scatter(0, 0, color="red", s=40)
# plt.text(0.1, 0.1, "global min", color="red")

# plt.title("Ackley function — filled contour")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


# # ---------------------------------------------------------
# # 4) 1D CROSS-SECTION (y = 0)
# # ---------------------------------------------------------
# xs = np.linspace(-5, 5, 600)
# ys = np.array([ackley([v, 0]) for v in xs])

# plt.figure(figsize=(12, 4))
# plt.plot(xs, ys, linewidth=2)

# plt.axvline(0, linestyle="--")
# plt.scatter(0, ackley([0, 0]), color="red", s=40)

# plt.title("Ackley slice along y = 0 (1D cross section)")
# plt.xlabel("x")
# plt.ylabel("f(x,0)")
# plt.grid(True)
# plt.show()
