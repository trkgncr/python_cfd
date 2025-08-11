'''
Created on 19 Eki 2016

@author: Turkay
'''
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation

a, b = 1.0, 1.0
c = np.pi / 2
u0 = 100
nx = 21
ny = 21
dx = a / (nx - 1)
dy = b / (ny - 1)
x = np.arange(0, a + dx, dx)
y = np.arange(0, b + dy, dy)
nt = 100
dt = 0.002

kx, ky = c ** 2 * dt / dx ** 2, c ** 2 * dt / dy ** 2

u = np.ones((nx, ny))
un = np.zeros((nx, ny))
u *= u0
u[0, :] = 0
u[nx - 1, :] = 0
u[:, 0] = 0
u[:, ny - 1] = 0

xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, u, rstride=1, cstride=1, color='b')
ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(11))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.00f'))
plt.title("Implicit Euler\nt = %.4f" % (0))

# fig.colorbar(surf, shrink=0.5, aspect=5)

A = np.zeros(((nx - 2) * (ny - 2), (nx - 2) * (ny - 2)))
B = np.zeros((nx - 2) * (ny - 2))


def index(i, j):
    return i + (nx - 2) * (j - 1)


def animate(k):
    un[:] = u[:]

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            A[index(i, j) - 1, index(i, j) - 1] = 2 * kx + 2 * ky + 1
            B[index(i, j) - 1] = un[i, j]
            if i == 1 and j == 1:  # sol-alt
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky # coef(0,3)
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx # coef(0,1)
            elif i == 1 and j == ny-2:  # sol-ust
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx
            elif i == nx-2 and j == 1:  # sag-alt
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx
            elif i == nx-2 and j == ny-2:  # sag-ust
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx
            elif j == 1:
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx
            elif j == ny-2:
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx
            elif i == 1:
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx
            elif i == nx-2:
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx
            else:
                A[index(i, j) - 1][index(i, j + 1) - 1] = -ky
                A[index(i, j) - 1][index(i, j - 1) - 1] = -ky
                A[index(i, j) - 1][index(i + 1, j) - 1] = -kx
                A[index(i, j) - 1][index(i - 1, j) - 1] = -kx

    u_temp = np.linalg.solve(A, B)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            u[i, j] = u_temp[index(i, j)-1]

    plt.cla()
    ax.plot_surface(xx, yy, u, rstride=1, cstride=1, color='b')
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.00f'))
    plt.title("Implicit Euler\nt = %.4f" % (k * dt))


# Init only required for blitting to give a clean slate.
def init():
    plt.cla()
    ax.plot_surface(xx, yy, u, rstride=1, cstride=1, color='b')
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.00f'))
    plt.title("Implicit Euler\nt = %.4f" % (0))


def iterate(t):
    cnt = 0
    while cnt < t:
        cnt += 1
        yield cnt


ani = animation.FuncAnimation(fig, animate, iterate(nt), init_func=init,
                              interval=50, blit=False, repeat='False')

plt.show()
