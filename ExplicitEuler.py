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
dt = 0.0002

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
plt.title("Explicit Euler\nt = %.4f" % (0))


# fig.colorbar(surf, shrink=0.5, aspect=5)

def animate(k):
    # for t in range(nt):
    un[:] = u[:]
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[i, j] = un[i, j] + (c ** 2) * dt * ((un[i + 1, j] - 2 * un[i, j] + un[i - 1, j]) / dx ** 2 + \
                                                  (un[i, j + 1] - 2 * un[i, j] + un[i, j - 1]) / dy ** 2)
    plt.cla()
    ax.plot_surface(xx, yy, u, rstride=1, cstride=1, color='b')
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.00f'))
    plt.title("Explicit Euler\nt = %.4f" % (k * dt))


# Init only required for blitting to give a clean slate.
def init():
    plt.cla()
    ax.plot_surface(xx, yy, u, rstride=1, cstride=1, color='b')
    ax.set_zlim(0, 100)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.00f'))
    plt.title("Explicit Euler\nt = %.4f" % (0))


def iterate(t):
    cnt = 0
    while cnt < t:
        cnt += 1
        yield cnt


ani = animation.FuncAnimation(fig, animate, iterate(nt), init_func=init,
                              interval=50, blit=False, repeat='False')

plt.show()
