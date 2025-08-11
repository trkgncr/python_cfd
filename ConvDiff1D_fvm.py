#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:17:40 2018

@author: turkay
"""
from math import exp
import matplotlib.pyplot as plt
import numpy as np

u = 1.0
rho = 1.0
gamma = 0.02

phi_0 = 0.0
phi_L = 1.0

L = 1.0
nx = 10
dx = L/nx
x_ref = np.linspace(0, L, nx+1)
phi_ref = np.zeros(nx+1)

"""
Analytical Solution
"""
def Q(x):
    return phi_0 + (exp(rho*u*x/gamma) - 1)/(exp(rho*u*L/gamma) - 1)*(phi_L-phi_0)

index = 0
for i in x_ref:
    phi_ref[index] = Q(i)
    index = index + 1


"""
Finite Volume Solution
"""
ite = 100
eps = 0.0001
orf = 1.2
#1 for Gauss-Seidel
#2 for SOR
solver = 1

cx = np.linspace(-dx/2, L+dx/2, nx+2)
fx = np.linspace(0, L, nx+1)
cx[0] = 0
cx[nx+1] = L
phi = np.zeros(nx+2)
Ae = np.zeros(nx)
Aw = np.zeros(nx)
An = np.zeros(nx)
As = np.zeros(nx)
Ap = np.zeros(nx)
b = np.zeros(nx)

phi[0] = phi_0
phi[nx+1] = phi_L
#Q[nx+1] = Q[nx+1] - phie*Fe - phie*AeD
rsm = np.zeros(ite)

for k in range(1, ite):
    phi_n = np.copy(phi)
#    for i in range(1, nx+1):
#        if i == 1:
#            Ae[i-1] = rho*u/2-gamma/dx
#            Aw[i-1] = -2*gamma/dx
#            Ap[i-1] = - (Ae[i-1] + Aw[i-1])
#            b[i-1] = rho*u
#        elif i == nx:
#            Ae[i-1] = -2*gamma/dx
#            Aw[i-1] = -rho*u/2-gamma/dx
#            Ap[i-1] = - (Ae[i-1] + Aw[i-1])
#            b[i-1] = -rho*u
#        else:
#            Ae[i-1] = rho*u/2-gamma/dx
#            Aw[i-1] = -rho*u/2-gamma/dx
#            Ap[i-1] = - (Ae[i-1] + Aw[i-1])
    for i in range(1, nx+1):
        Ae[i-1] = rho*u/2.0-gamma/(cx[i+1]-cx[i])
        Aw[i-1] = -rho*u/2.0-gamma/(cx[i]-cx[i-1])
        Ap[i-1] = - (Ae[i-1] + Aw[i-1])
        
        if solver == 1:
            phi[i] = (b[i-1]-Ae[i-1]*phi_n[i + 1] - Aw[i-1]*phi_n[i - 1]) / (Ap[i-1] + 1e-8)
        elif solver == 2:
            phi[i] = (orf*(b[i-1] -Aw[i-1]*phi[i - 1] - Ae[i-1]*phi_n[i + 1]) / (Ap[i-1] + 1e-8) 
                    + (1 - orf)*phi_n[i])

    b[0] = b[0] + phi[0]*rho*u + phi[0]*gamma/(cx[1] - cx[0])
    b[nx-1] = b[nx-1] - phi[nx+1]*rho*u + phi[nx+1]*gamma/(cx[nx+1] - cx[nx])
    
    Ap[0] = Ap[0] + rho*u + gamma/(cx[nx+1] - cx[nx])
    Ap[nx] = Ap[nx] - rho*u + gamma/(cx[nx+1] - cx[nx])
    
    sum1, sum2 = 0, 0
    for i in range(1, nx+1):
        sum1 += abs(-Ae[i-1]*phi[i+1] - Aw[i-1]*phi[i-1] - Ap[i-1]*phi[i])
        sum2 += abs(Ap[i-1]*phi[i])
    rsm[k] = sum1/(sum2+1e-8)
    if rsm[k] < eps:
        break
    
plt.figure()
plt.title("Finite Volume Solution")
plt.xlabel("x-axis")
plt.ylabel("phi")
plt.plot(x_ref, phi_ref, 'b-', cx, phi, 'r.')
plt.show()