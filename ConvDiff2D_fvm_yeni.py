import matplotlib.pyplot as plt
import numpy as np

u = 1.0
rho = 1.0
gamma = 0.01

phi_U = 0.0
#phi_B = 0.0
#phi_L = 0.0
#phi_R = 0.0

L = 1.0
W = 1.0
nx = 50
ny = 50
dx = L/nx
dy = W/ny

Pe = rho * u * dx / gamma


"""
Finite Volume Solution
"""
ite = 1000
eps = 0.0001
orf = 1.2
#1 for Gauss-Seidel
#2 for SOR
solver = 1

cx = np.linspace(-dx/2, L+dx/2, nx+2)
cx[0] = 0
cx[nx+1] = L

cy = np.linspace(-dy/2, W+dy/2, ny+2)
cy[0] = 0
cy[ny+1] = W

phi = np.zeros((nx+2, ny +2))
Ae = np.zeros((nx, ny))
Aw = np.zeros((nx, ny))
An = np.zeros((nx, ny))
As = np.zeros((nx, ny))
Ap = np.zeros((nx, ny))
b = np.zeros((nx, ny))

"""
Boundary Conditions
"""
phi[:, ny+1] = phi_U

for j in range(ny+2):
    phi[0, j] = -cy[j] + 1

rsm = np.zeros(ite)

for i in range(1, nx+1):
    for j in range(1, ny+1):
        Ae[i-1, j-1] = rho*cx[i]*dy/2.0-gamma*dy/(cx[i+1]-cx[i])
        Aw[i-1, j-1] = -rho*cx[i]*dy/2.0-gamma*dy/(cx[i]-cx[i-1])
        An[i-1, j-1] = -rho*cy[j]*dx/2.0-gamma*dx/(cx[j+1]-cx[j])
        As[i-1, j-1] = rho*cy[j]*dx/2.0-gamma*dx/(cx[j]-cx[j-1])
        Ap[i-1, j-1] = - (Ae[i-1, j-1] + Aw[i-1, j-1] + An[i-1, j-1] + As[i-1, j-1])

#b[0] = b[0] + phi[0]*rho*u + phi[0]*gamma/(cx[1] - cx[0])
#b[:,ny-1] = b[:,ny-1] - phi[:,ny+1]*rho*u + phi[:,ny+1]*gamma/(cx[nx+1] - cx[nx])

#Ap[0] = Ap[0] + rho*u/2.0
#Aw[0] = 0
#Ap[nx-1] = Ap[nx-1] - rho*u/2.0
#Ae[nx-1] = 0

for k in range(1, ite):
    phi_n = np.copy(phi)

    for i in range(1, nx+1): 
        for j in range(1, ny+1):
            if solver == 1:
                phi[i, j] = (b[i-1, j-1] - Ae[i-1, j-1]*phi_n[i+1, j] - Aw[i-1, j-1]*phi_n[i-1, j] - An[i-1, j-1]*phi_n[i, j+1] - As[i-1, j-1]*phi_n[i, j-1]) / (Ap[i-1, j-1] + 1e-8)
            elif solver == 2:
                phi[i, j] = (orf*(b[i-1, j-1] - Aw[i-1, j-1]*phi[i-1, j] - Ae[i-1, j-1]*phi_n[i+1, j] - As[i-1, j-1]*phi[i, j-1] - An[i-1, j-1]*phi_n[i, j+1]) / (Ap[i-1, j-1] + 1e-8) 
                    + (1 - orf)*phi_n[i, j])
    
    phi[:, 0] = phi[:, 1]
    phi[nx+1, :] = phi[nx, :]   
    
    sum1, sum2 = 0, 0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            sum1 += abs(-Ae[i-1, j-1]*phi[i+1, j] - Aw[i-1, j-1]*phi[i-1, j] - An[i-1, j-1]*phi[i, j+1] - As[i-1, j-1]*phi[i, j-1] - Ap[i-1, j-1]*phi[i, j])
            sum2 += abs(Ap[i-1, j-1]*phi[i, j])
    rsm[k] = sum1/(sum2+1e-8)
    print("iter: ", k)
    if rsm[k] < eps:
        break
 
xx, yy = np.meshgrid(cx, cy, sparse=False, indexing='ij')    
plt.figure()
plt.title("Finite Difference Solution")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
CS = plt.contour(xx, yy, phi)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()