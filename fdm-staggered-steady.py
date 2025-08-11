#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 26 Kas 2016

@author: Turkay
'''

import numpy as np
import matplotlib.pyplot as plt

solver = 2
nx = 50
ny = 50
#nt = 9
#dt = 0.01
n_it = 1000
n_ot = 1000
mu = 0.01
rho = 1.0
nu = mu / rho
L = 1.0
W = 1.0
dx = L / nx
dy = W / ny
alphaP = 0.3
alphaM = 0.7
orf = 1.2
eps = 0.001

x = np.linspace(0, L, nx)
y = np.linspace(0, W, ny)

u = np.zeros((nx+1, ny+2), dtype=np.float)
uTemp = np.zeros((nx+1, ny+2), dtype=np.float)
uCorr = np.zeros((nx+1, ny+2), dtype=np.float)
um = np.zeros((nx+1, ny+2), dtype=np.float)
#un = np.zeros((nx+1, ny+2), dtype=np.float)

v = np.zeros((nx+2, ny+1), dtype=np.float)
vTemp = np.zeros((nx+2, ny+1), dtype=np.float)
vCorr = np.zeros((nx+2, ny+1), dtype=np.float)
vm = np.zeros((nx+2, ny+1), dtype=np.float)
#vn = np.zeros((nx+2, ny+1), dtype=np.float)

p = np.zeros((nx+2, ny+2), dtype=np.float)
pTemp = np.zeros((nx+2, ny+2), dtype=np.float)
pCorr = np.zeros((nx+2, ny+2), dtype=np.float)
pm = np.zeros((nx+2, ny+2), dtype=np.float)
#pn = np.zeros((nx+2, ny+2), dtype=np.float)

# Initial Conditions
uLid = 1.0

rsm_uCorr = np.zeros(n_ot)
rsm_vCorr = np.zeros(n_ot)
rsm_pCorr = np.zeros(n_ot)

aP = 2.0 * nu / dx**2.0 + 2.0 * nu / dy**2.0
aEp = 1.0 / aP / dx**2.0
aWp = 1.0 / aP / dx**2.0
aNp = 1.0 / aP / dy**2.0
aSp = 1.0 / aP / dy**2.0
aPp = - (aEp + aWp + aNp + aSp)

for m in range(n_ot):
    print("\niter:", m+1)
    # Set Dirichlet BCs for u and v
    v[0, :] = - v[1, :]  # West BC
    v[nx + 1, :] = - v[nx, :]  # East BC
    u[:, 0] = - u[:, 1]  # South BC
    u[:, ny + 1] = 2.0 * uLid - u[:, ny]  # North BC

    um[:, :] = u[:, :]
    vm[:, :] = v[:, :]
    rsm_u = np.zeros(n_it)
    rsm_v = np.zeros(n_it)

    for k in range(n_it):
        uTemp[:,:] = u[:,:]
        vTemp[:,:] = v[:,:]
            
        ApStar = aP / alphaM

#Solve u---------------------------
        for i in range(1, nx):
            for j in range(1, ny + 1):
                vInt = (vm[i, j - 1] + vm[i, j] + vm[i + 1, j - 1] + vm[i + 1, j]) / 4.0
                Ae = um[i, j] / (2.0 * dx) - nu / dx ** 2
                Aw = -um[i, j] / (2.0 * dx) - nu / dx ** 2
                An = vInt / (2.0 * dy) - nu / dy ** 2
                As = -vInt / (2.0 * dy) - nu / dy ** 2
                Qu = -(p[i + 1, j] - p[i, j]) / dx
                QuStar = Qu + (1. - alphaM) / alphaM * aP * um[i, j]
                    
                if solver == 1:
                    u[i, j] = 1 / ApStar * (
                            QuStar - (Ae * uTemp[i + 1, j] + Aw * uTemp[i - 1, j] + An * uTemp[i, j + 1] + As * uTemp[i, j - 1]))
                elif solver == 2:
                    u[i, j] = orf / ApStar * (
                        QuStar - (Ae * uTemp[i + 1, j] + Aw * u[i - 1, j] + An * uTemp[i, j + 1] + As * u[i, j - 1])) + \
                            (1. - orf) * uTemp[i, j]

        for i in range(1, nx):
            for j in range(1, ny + 1):
                rsm_u[k] += abs((u[i, j] - uTemp[i, j]) / (uTemp[i, j] + 1e-8))

#Solve v------------------------------------------
        for i in range(1, nx + 1):
            for j in range(1, ny):
                uInt = (um[i - 1, j] + um[i - 2, j + 1] + um[i, j] + um[i, j + 1]) / 4.0
                Ae = uInt / (2.0 * dx) - nu / dx ** 2
                Aw = -uInt / (2.0 * dx) - nu / dx ** 2
                An = vm[i][j] / (2.0 * dy) - nu / dy**2.0
                As = -vm[i][j] / (2.0 * dy) - nu / dy**2.0
                Qv = -(p[i, j + 1] - p[i, j]) / dy
                QvStar = Qv + (1. - alphaM) / alphaM * aP * vm[i, j]

                if solver == 1:
                    v[i, j] = 1. / ApStar * (
                        QvStar - (Ae * vTemp[i + 1, j] + Aw * vTemp[i - 1, j] + An * vTemp[i, j + 1] + As * vTemp[i, j - 1]))
                elif solver == 2:
                    v[i, j] = orf / ApStar * (
                        QvStar - (Ae * vTemp[i + 1, j] + Aw * v[i - 1, j] + An * vTemp[i, j + 1] + As * v[i, j - 1])) + \
                            (1. - orf) * vTemp[i, j]

        for i in range(1, nx + 1):
            for j in range(1, ny):
                rsm_v[k] += abs((v[i, j] - vTemp[i, j]) / (vTemp[i, j] + 1e-8))
#
        if rsm_u[k] < eps and rsm_v[k] < eps:
            print("inner ite vel: ", k+1)
            break

    rsm_p = np.zeros(n_it)
#Solve p'----------------------------------------------------
    pCorr[1,1] = 0 #Reference pressure
        
    pCorr[0, :] = pCorr [1, :] # West BC
    pCorr[nx + 1, :] = pCorr [nx, :] # East BC
    pCorr[:, 0] = pCorr[:, 1] # South BC
    pCorr[:, ny + 1] = pCorr[:, ny] # North BC
        
    for k in range(n_it):
        pTemp[:,:] = pCorr[:,:]
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                if i != 1 and j != 1:
                    Qp = (u[i, j] - u[i - 1, j]) / dx + (v[i, j] - v[i, j - 1]) / dy
                    if solver == 1:
                        pCorr[i, j] = (Qp - aEp * pTemp[i + 1, j] - aWp * pTemp[i - 1, j] - aNp * 
                             pTemp[i, j + 1] - aSp * pTemp[i, j - 1]) / aPp
                    elif solver == 2:
                        pCorr[i, j] = orf / aPp * (
                            Qp - (aEp * pTemp[i + 1, j] + aWp * pCorr[i - 1, j] + aNp * pTemp[i, j + 1] + aSp * pCorr[i, j - 1])) + \
                                (1. - orf) * pTemp[i, j]

        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                rsm_p[k] += abs((pCorr[i, j] - pTemp[i, j]) / (pTemp[i, j]+1e-16))
                    
        if rsm_p[k] < eps:
            print("inner ite press: ", k+1)
            break
#Correction-------------------------------------------------
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            p[i, j] += pCorr[i, j] * alphaP
    rsm_pCorr[m] = abs(np.amax(pCorr))
                
    for i in range(1, nx):
        for j in range(1, ny + 1):
            uCorr[i, j] = - (pCorr[i + 1, j] - pCorr[i, j]) / aP / dx
            u[i, j] += uCorr[i, j]
    rsm_uCorr[m] = abs(np.amax(uCorr))
                
    for i in range(1, nx + 1):
        for j in range(1, ny):        
            vCorr[i, j] = - (pCorr[i, j + 1] - pCorr[i, j]) / aP / dy
            v[i, j] += vCorr[i, j]
    rsm_vCorr[m] = abs(np.amax(vCorr))

        # for j in range(2, ny - 2):
        #     p[0, j] = p[2, j]  # West BC
        #     p[nx - 1, j] = p[nx - 3, j]  # East BC
        #
        # for i in range(2, nx - 2):
        #     p[i, 0] = p[i, 2]  # South BC
        #     p[i, ny - 1] = p[i, ny - 3]  # North BC

#        if rsm_uCorr[m] < eps and rsm_vCorr[m] < eps:
#            print("Outer iteration converged!")
#            break


#xx, yy = np.meshgrid(x, y, sparse=False, indexing='ij')

plt.figure(1)
# CS = plt.contourf(xx, yy, uAbs)
# plt.clabel(CS, inline=1, fontsize=10)
# plt.title("Finite Difference Solution\nCavity Flow")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#plt.figure(1)
#ite1 = np.arange(1, n_it + 1)
#ite2 = np.arange(1, n_ot + 1)
## Three subplots sharing both x/y axes
#plt.semilogy(ite2, rsm_uCorr, 'r', ite2, rsm_vCorr, 'b', ite2, rsm_pCorr, 'g')
#plt.figure(2)
ux = np.zeros(ny, dtype=np.float)
for i in range(ny):
    ux[i] = u[nx/2,i]
    
plt.plot(ux, range(ny))

plt.show()
