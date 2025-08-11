#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 26 Kas 2016

@author: Turkay
'''

import numpy as np
import matplotlib.pyplot as plt

solver = 2
nx = 10
ny = 10
nt = 200
n_it = 5
n_ot = 20
dt = 0.01
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
un = np.zeros((nx+1, ny+2), dtype=np.float)

v = np.zeros((nx+2, ny+1), dtype=np.float)
vTemp = np.zeros((nx+2, ny+1), dtype=np.float)
vCorr = np.zeros((nx+2, ny+1), dtype=np.float)
vm = np.zeros((nx+2, ny+1), dtype=np.float)
vn = np.zeros((nx+2, ny+1), dtype=np.float)

p = np.zeros((nx+2, ny+2), dtype=np.float)
pTemp = np.zeros((nx+2, ny+2), dtype=np.float)
pCorr = np.zeros((nx+2, ny+2), dtype=np.float)
pm = np.zeros((nx+2, ny+2), dtype=np.float)
pn = np.zeros((nx+2, ny+2), dtype=np.float)

aEu = np.zeros((nx + 1, ny + 2))
aWu = np.zeros((nx + 1, ny + 2))
aNu = np.zeros((nx + 1, ny + 2))
aSu = np.zeros((nx + 1, ny + 2))
aPu = np.zeros((nx + 1, ny + 2))
aEv = np.zeros((nx + 2, ny + 1))
aWv = np.zeros((nx + 2, ny + 1))
aNv = np.zeros((nx + 2, ny + 1))
aSv = np.zeros((nx + 2, ny + 1))
aPv = np.zeros((nx + 2, ny + 1))
aEp = np.zeros((nx + 2, ny + 2))
aWp = np.zeros((nx + 2, ny + 2))
aNp = np.zeros((nx + 2, ny + 2))
aSp = np.zeros((nx + 2, ny + 2))
aPp = np.zeros((nx + 2, ny + 2))

# Initial Conditions
uLid = 1.0

rsm_uCorr = np.zeros(n_ot)
rsm_vCorr = np.zeros(n_ot)
rsm_pCorr = np.zeros(n_ot)

#aP = 2.0 * nu / dx**2.0 + 2.0 * nu / dy**2.0 + 1.0 / dt
#aEp = 1.0 / aP / dx**2.0
#aWp = 1.0 / aP / dx**2.0
#aNp = 1.0 / aP / dy**2.0
#aSp = 1.0 / aP / dy**2.0
#aPp = - (aEp + aWp + aNp + aSp)

plt.ion()
plt.figure()
ax = plt.gca()
ux = np.zeros(ny+2, dtype=np.float)
half = int(ny/2)
y = np.linspace(-dy/2.0, 1+dy/2.0, num=ny+2)
ux_ref = [0.0000, -0.03717, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.84123, 1.00000]
y_ref = [0.0000, 0.0547, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9766, 1.0000]

for n in range(nt):
    print("\n\nTime step: ", n+1)

    un[:, :] = u[:, :]
    vn[:, :] = v[:, :]
    pn[:, :] = p[:, :]

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
            
#            ApStar = aP / alphaM

#Solve u---------------------------
            for i in range(1, nx):
                for j in range(1, ny + 1):
                    vInt = (vm[i, j - 1] + vm[i, j] + vm[i + 1, j - 1] + vm[i + 1, j]) / 4.0
                    aPu[i, j] = 2.0 * nu / dx**2.0 + 2.0 * nu / dy**2.0 + 1.0 / dt
                    ApStar = aPu[i,j] / alphaM
                    aEu[i, j] = um[i, j] / (2.0 * dx) - nu / dx ** 2
                    aWu[i, j] = -um[i, j] / (2.0 * dx) - nu / dx ** 2
                    aNu[i, j] = vInt / (2.0 * dy) - nu / dy ** 2
                    aSu[i, j] = -vInt / (2.0 * dy) - nu / dy ** 2
                    Qu = un[i, j] / dt - (p[i + 1, j] - p[i, j]) / dx
                    QuStar = Qu + (1. - alphaM) / alphaM * aPu[i,j] * u[i, j]
                    
                    if solver == 1:
                        u[i, j] = 1 / ApStar * (
                                QuStar - (aEu[i, j] * uTemp[i + 1, j] + aWu[i, j] * uTemp[i - 1, j] + aNu[i, j] * uTemp[i, j + 1] + aSu[i, j] * uTemp[i, j - 1]))
                    elif solver == 2:
                        u[i, j] = orf / ApStar * (
                            QuStar - (aEu[i, j] * uTemp[i + 1, j] + aWu[i, j] * u[i - 1, j] + aNu[i, j] * uTemp[i, j + 1] + aSu[i, j] * u[i, j - 1])) + \
                                  (1. - orf) * uTemp[i, j]

            for i in range(1, nx):
                for j in range(1, ny + 1):
                    rsm_u[k] += abs((u[i, j] - uTemp[i, j]) / (uTemp[i, j] + 1e-8))

#Solve v------------------------------------------
            for i in range(1, nx + 1):
                for j in range(1, ny):
                    uInt = (um[i - 1, j] + um[i - 2, j + 1] + um[i, j] + um[i, j + 1]) / 4.0
                    aPv[i, j] = 2.0 * nu / dx**2.0 + 2.0 * nu / dy**2.0 + 1.0 / dt
                    ApStar = aPv[i,j] / alphaM
                    aEv[i, j] = uInt / (2.0 * dx) - nu / dx ** 2
                    aWv[i, j] = -uInt / (2.0 * dx) - nu / dx ** 2
                    aNv[i, j] = vm[i][j] / (2.0 * dy) - nu / dy**2.0
                    aSv[i, j] = -vm[i][j] / (2.0 * dy) - nu / dy**2.0
                    Qv = vn[i, j] / dt - (p[i, j + 1] - p[i, j]) / dy
                    QvStar = Qv + (1. - alphaM) / alphaM * aPv[i, j] * v[i, j]

                    if solver == 1:
                        v[i, j] = 1. / ApStar * (
                                QvStar - (aEv[i, j] * vTemp[i + 1, j] + aWv[i, j] * vTemp[i - 1, j] + aNv[i, j] * vTemp[i, j + 1] + aSv[i, j] * vTemp[i, j - 1]))
                    elif solver == 2:
                        v[i, j] = orf / ApStar * (
                            QvStar - (aEv[i, j] * vTemp[i + 1, j] + aWv[i, j] * v[i - 1, j] + aNv[i, j] * vTemp[i, j + 1] + aSv[i, j] * v[i, j - 1])) + \
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
        
#        pCorr[0, :] = pCorr [1, :] # West BC
#        pCorr[nx + 1, :] = pCorr [nx, :] # East BC
#        pCorr[:, 0] = pCorr[:, 1] # South BC
#        pCorr[:, ny + 1] = pCorr[:, ny] # North BC
        
        for k in range(n_it):
            pTemp[:,:] = pCorr[:,:]
            for i in range(1, nx + 1):
                for j in range(1, ny + 1):
                    aEp[i, j] = -1.0 / (aPu[i, j] + 1e-8) / dx
                    aWp[i, j] = -1.0 / (aPu[i - 1, j] + 1e-8) / dx
                    aNp[i, j] = -1.0 / (aPv[i, j] + 1e-8) / dy
                    aSp[i, j] = -1.0 / (aPv[i, j - 1] + 1e-8) / dy
#                    aEp[nx, :] = 0.0 # Dogu
#                    aWp[1, :] = 0.0 # Bati
#                    aNp[:, ny] = 0.0 # Kuzey
#                    aSp[:, 1] = 0.0 # Guney
                    aPp[i, j] = - (aEp[i, j] + aWp[i, j] + aNp[i, j] + aSp[i, j])
#                   Qp = (u[i, j] - u[i - 1, j]) / dx + (v[i, j] - v[i, j - 1]) / dy
                    Qp = - (u[i, j] - u[i - 1, j] + v[i, j] - v[i, j - 1])
                    if solver == 1:
                        pCorr[i, j] = (Qp - aEp[i, j] * pTemp[i + 1, j] - aWp[i, j] * pTemp[i - 1, j] - aNp[i, j] * 
                             pTemp[i, j + 1] - aSp[i, j] * pTemp[i, j - 1]) / aPp[i, j]
                    elif solver == 2:
                        pCorr[i, j] = orf / aPp[i, j] * (
                            Qp - (aEp[i, j] * pTemp[i + 1, j] + aWp[i, j] * pCorr[i - 1, j] + aNp[i, j] * pTemp[i, j + 1] + aSp[i, j] * pCorr[i, j - 1])) + \
                                (1. - orf) * pTemp[i, j]
            pCorr[0,0] = 0.0

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
                uCorr[i, j] = - (pCorr[i + 1, j] - pCorr[i, j]) / aPu[i, j] / dx
                u[i, j] += uCorr[i, j]
        rsm_uCorr[m] = abs(np.amax(uCorr))
                
        for i in range(1, nx + 1):
            for j in range(1, ny):        
                vCorr[i, j] = - (pCorr[i, j + 1] - pCorr[i, j]) / aPv[i, j] / dy
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



        for j in range(ny+2):
            ux[j] = u[half,j]
        ux[0] = 0
        ux[ny+1] = 1
        
        ax.clear()
        ax.plot(ux, y, 'r--', ux_ref, y_ref, 'bs')
        plt.draw()
        plt.pause(0.01)
    

