# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time
start_time = time.time()

L = 1 # Kare oyuk
W = 1
Nx = 50 # x ve y yonunde hucre sayisi (ard-işlem için çift sayı tercih edilmeli.)
Ny = 50
d = L / Nx # uniform ag boyutu
ULid = 1.0
nu = 0.01
dt = 0.01
tMax = 2.0
noOfTSteps = int(tMax / dt)
maxOuterIt = 50
#maxInnerIt = 200
epsU = 1.e-3
epsV = 1.e-3
epsP = 1.e-3
alpha = 0.8
alphaP = 0.3
omg = 1.2 # SOR katsayisi
M = 10 # p' hata normalizasyonu icin

cx = np.linspace(-d/2, L+d/2, Nx+2)
cx[0] = 0
cx[Nx+1] = L

fx = np.linspace(0, L, Nx+1)

cy = np.linspace(-d/2, W+d/2, Ny+2)
cy[0] = 0
cy[Ny+1] = W

fy = np.linspace(0, W, Ny+1)

u = np.zeros((Nx + 1, Ny + 2))
v = np.zeros((Nx + 2, Ny + 1))
p = np.zeros((Nx + 2, Ny + 2))
uStar = np.zeros((Nx + 1, Ny + 2))
vStar = np.zeros((Nx + 2, Ny + 1))
pStar = np.zeros((Nx + 2, Ny + 2))
uCorr = np.zeros((Nx + 1, Ny + 2))
vCorr = np.zeros((Nx + 2, Ny + 1))
pCorr = np.zeros((Nx + 2, Ny + 2))
uOld = np.zeros((Nx + 1, Ny + 2))
vOld = np.zeros((Nx + 2, Ny + 1))
pCorrOld = np.zeros((Nx + 2, Ny + 2))
#aEu = np.zeros((N + 1, N + 2))
#aWu = np.zeros((N + 1, N + 2))
#aNu = np.zeros((N + 1, N + 2))
#aSu = np.zeros((N + 1, N + 2))
aPu = np.zeros((Nx + 1, Ny + 2))
#aEv = np.zeros((N + 2, N + 1))
#aWv = np.zeros((N + 2, N + 1))
#aNv = np.zeros((N + 2, N + 1))
#aSv = np.zeros((N + 2, N + 1))
aPv = np.zeros((Nx + 2, Ny + 1))
#aEp = np.zeros((N + 2, N + 2))
#aWp = np.zeros((N + 2, N + 2))
#aNp = np.zeros((N + 2, N + 2))
#aSp = np.zeros((N + 2, N + 2))
aPp = np.zeros((Nx + 2, Ny + 2))

resNormP = np.zeros(M)
outerResNorm = np.zeros(M)
outerRes = np.zeros(maxOuterIt)

# U: Representation of sparse matrix and right-hand side
NU = (Nx+1)*(Ny+2)
mainU = np.ones(NU)               #diagonal
lowerU = np.zeros(NU-1)           #subdiagonal
upperU = np.zeros(NU-1)           #superdiagonal
lower2U = np.zeros(NU-(Nx+1))     #lower diagonal
upper2U = np.zeros(NU-(Nx+1))     #upper diagonal
bU = np.zeros(NU)                 #right-hand side

# V: Representation of sparse matrix and right-hand side
NV = (Nx+2)*(Ny+1)
mainV = np.ones(NV)               #diagonal
lowerV = np.zeros(NV-1)           #subdiagonal
upperV = np.zeros(NV-1)           #superdiagonal
lower2V = np.zeros(NV-(Nx+1))     #lower diagonal
upper2V = np.zeros(NV-(Nx+1))     #upper diagonal
bV = np.zeros(NV)                 #right-hand side

# P': Representation of sparse matrix and right-hand side
NP = (Nx+2)*(Ny+2)
mainP = np.ones(NP)               #diagonal
lowerP = np.zeros(NP-1)           #subdiagonal
upperP = np.zeros(NP-1)           #superdiagonal
lower2P = np.zeros(NP-(Nx+1))     #lower diagonal
upper2P = np.zeros(NP-(Nx+1))     #upper diagonal
bP = np.zeros(NP)                 #right-hand side

# Zaman dongusunu baslat

for tStep in range(1, noOfTSteps + 1):
    t = tStep * dt
    print("\n\t\tTIME STEP: %d TIME = %.3f s" %(tStep, t))
    
# u,v ve p icin bir onceki zamana ait cozumleri kopyala    
    uOld = np.copy(u)
    vOld = np.copy(v)
    uStar = np.copy(u)
    vStar = np.copy(v)
    pStar = np.copy(p)

# A katsayilarini bir onceki zamana ait hizlar ile hesapla

# u denklemi katsayilari

#    for i in range(1, N):
#        for j in range(1, N + 1):
#            aEu[i][j] =   0.5 * 0.5 * (u[i][j] + u[i + 1][j]) * d - nu 
#            aWu[i][j] = - 0.5 * 0.5 * (u[i - 1][j] + u[i][j]) * d - nu
#            aNu[i][j] =   0.5 * 0.5 * (v[i][j] + v[i + 1][j]) * d - nu
#            aSu[i][j] = - 0.5 * 0.5 * (v[i][j - 1] + v[i + 1][j - 1]) * d - nu
#            aPu[i][j] =   aEu[i][j] + aWu[i][j] + aNu[i][j] + aSu[i][j] + 8.0 * nu + d**2.0 / dt

# v denklemi katsayilari

#    for i in range(1, N + 1):
#        for j in range(1, N):
#            aEv[i][j] =   0.5 * 0.5 * (u[i][j] + u[i][j + 1]) * d - nu 
#            aWv[i][j] = - 0.5 * 0.5 * (u[i - 1][j] + u[i - 1][j + 1]) * d - nu
#            aNv[i][j] =   0.5 * 0.5 * (v[i][j] + v[i][j + 1]) * d - nu
#            aSv[i][j] = - 0.5 * 0.5 * (v[i][j - 1] + v[i][j]) * d - nu
#            aPv[i][j] =   aEv[i][j] + aWv[i][j] + aNv[i][j] + aSv[i][j] + 8.0 * nu + d**2.0 / dt 

# p' denklemi katsayilari

#    for i in range(1, N + 1):
#        for j in range(1, N + 1): 
#            aEp[i][j] = - d / (aPu[i][j] + 1e-8)
#            aWp[i][j] = - d / (aPu[i - 1][j] + 1e-8)
#            aNp[i][j] = - d / (aPv[i][j] + 1e-8) 
#            aSp[i][j] = - d / (aPv[i][j - 1] + 1e-8)

#    aEp[N, :] = 0.0 # Dogu
#    aWp[1, :] = 0.0 # Bati
#    aNp[:, N] = 0.0 # Kuzey
#    aSp[:, 1] = 0.0 # Guney

#    aPp =   - ( aEp + aWp + aNp + aSp )
                
# Dis iterasyon dongusunu baslat

    for m in range(1, maxOuterIt + 1):
        print("\nOuter iteration: %d" %(m))

# u ve v icin sinir kosullarini uygula

#        vStar[0, :] = - vStar[1, :] # Bati
#        vStar[N + 1, :] = - vStar[N, :] # Dogu
#        uStar[:, 0] = - uStar[:, 1] # Guney
#        uStar[:, N + 1] = 2.0 * ULid - uStar[:, N] # Kuzey    
        
# u* cozumu yap        

        for i in range(1, Nx):
            for j in range(1, Ny + 1):
                h = j*(Nx+1) + i
                aEu =   0.5 * 0.5 * (u[i,j] + u[i+1,j]) * (fy[j]-fy[j-1]) - nu * (fy[j]-fy[j-1])/(fx[i+1]-fx[i])
                aWu = - 0.5 * 0.5 * (u[i-1,j] + u[i,j]) * (fy[j]-fy[j-1]) - nu * (fy[j]-fy[j-1])/(fx[i]-fx[i-1])
                aNu =   0.5 * 0.5 * (v[i,j] + v[i+1,j]) * (cx[i+1]-cx[i]) - nu * (cx[i+1]-cx[i])/(cy[j+1]-cy[j])
                aSu = - 0.5 * 0.5 * (v[i,j-1] + v[i+1,j-1]) * (cx[i+1]-cx[i]) - nu * (cx[i+1]-cx[i])/(cy[j]-cy[j-1])
                aPu[i,j] = -(aEu + aWu + aNu + aSu) + d**2 / dt
                    
                aPRelax = aPu[i,j] / alpha 
                QuAug = uOld[i,j] * d**2.0 / dt - (pStar[i+1,j] - pStar[i,j]) * d
                QuAugRelax = QuAug + (1 - alpha) / alpha * aPu[i,j] * u[i,j]
                
                mainU[h] = aPRelax
                upperU[h] = aEu
                lowerU[h-1] = aWu
                upper2U[h] = aNu
                lower2U[h-(Nx+2)] = aSu                
                bU[h] = QuAugRelax
                    
        # Bottom
        for i in range(Nx+1):
            h = 0*(Nx+1) + i
            bU[h] = 0
    
        # Right
        for j in range(Ny+2):
            h = j*(Nx+1) + (Nx)
            bU[h] = 0
    
        # Left
        for j in range(Ny+2):
            h = j*(Nx+1) + 0
            bU[h] = 0
    
        # Up
        for i in range(Nx+1):
            h = (Ny+1)*(Nx+1) + i
            bU[h] = ULid            
                    
        diagonals = [mainU, upperU, lowerU, upper2U, lower2U]
        A = diags(diagonals, [0, 1, -1, Nx+2, -(Nx+2)])
        
        # Solve matrix system A*c = b
        c = scipy.sparse.linalg.spsolve(A, bU)

        # Fill u with vector c
        uStar[:,:] = c.reshape(Ny+2,Nx+1).T
            
        
# v* cozumu yap        
        
        for i in range(1, Nx + 1):
            for j in range(1, Ny):
                h = j*(Nx+2) + i
                aEv =   0.5 * 0.5 * (u[i,j] + u[i,j+1]) * (cy[j+1]-cy[j]) - nu * (cy[j+1]-cy[j])/(cx[i+1]-cx[i])
                aWv = - 0.5 * 0.5 * (u[i-1,j] + u[i-1,j+1]) * (cy[j+1]-cy[j]) - nu * (cy[j+1]-cy[j])/(cx[i]-cx[i-1])
                aNv =   0.5 * 0.5 * (v[i,j] + v[i,j+1]) * (fx[i]-fx[i-1]) - nu * (fx[i]-fx[i-1])/(fy[j+1]-fy[j])
                aSv = - 0.5 * 0.5 * (v[i,j] + v[i,j-1]) * (fx[i]-fx[i-1]) - nu * (fx[i]-fx[i-1])/(fy[j]-fy[j-1])
                aPv[i,j] = -(aEv + aWv + aNv + aSv) + d**2 / dt
                
                aPRelax = aPv[i,j] / alpha 
                QvAug = vOld[i,j] * d**2.0 / dt - (pStar[i,j+1] - pStar[i,j]) * d
                QvAugRelax = QvAug + (1 - alpha) / alpha * aPv[i,j] * v[i,j]
                
                mainV[h] = aPRelax
                upperV[h] = aEv
                lowerV[h-1] = aWv
                upper2V[h] = aNv
                lower2V[h-(Nx+2)] = aSv                
                bV[h] = QvAugRelax
                
        # Bottom
        for i in range(Nx+2):
            h = 0*(Nx+2) + i
            bV[h] = 0
    
        # Right
        for j in range(Ny+1):
            h = j*(Nx+2) + (Nx+1)
            bV[h] = 0
    
        # Left
        for j in range(Ny+1):
            h = j*(Nx+2) + 0
            bV[h] = 0
    
        # Up
        for i in range(Nx+2):
            h = (Ny)*(Nx+2) + i
            bV[h] = 0            
                    
        diagonals = [mainV, upperV, lowerV, upper2V, lower2V]
        A = diags(diagonals, [0, 1, -1, Nx+2, -(Nx+2)])
        
        # Solve matrix system A*c = b
        c = scipy.sparse.linalg.spsolve(A, bV)

        # Fill u with vector c
        vStar[:,:] = c.reshape(Ny+1,Nx+2).T
             
# p' icin cozum yap
        pCorrOld = np.copy(pCorr)
            
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                h = j*(Nx+2) + i
                aEp = - d / (aPu[i][j] + 1e-8)
                aWp = - d / (aPu[i - 1][j] + 1e-8)
                aNp = - d / (aPv[i][j] + 1e-8) 
                aSp = - d / (aPv[i][j - 1] + 1e-8)
                aPp[i][j] =   - ( aEp + aWp + aNp + aSp )
                
                Qp = - ( uStar[i][j] - uStar[i - 1][j] + vStar[i][j] - vStar[i][j - 1] )
                
                mainP[h] = aPp[i][j]
                upperP[h] = aEp
                lowerP[h-1] = aWp
                upper2P[h] = aNp
                lower2P[h-(Nx+2)] = aSp                
                bP[h] = Qp
                    
        diagonals = [mainP, upperP, lowerP, upper2P, lower2P]
        A = diags(diagonals, [0, 1, -1, Nx+2, -(Nx+2)])
        
        # Solve matrix system A*c = b
        c = scipy.sparse.linalg.spsolve(A, bP)

        # Fill u with vector c
        pCorr[:,:] = c.reshape(Ny+2,Nx+2).T
        
# p'yi duzelt

        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                pStar[i][j] = pStar[i][j] + pCorr[i][j] * alphaP         

# u' ve v' hesapla ve u ile v'yi duzelt

        for i in range(1, Nx):
            for j in range(1, Ny + 1): 
                uCorr[i][j] = - d / aPu[i][j] * (pCorr[i + 1][j] - pCorr[i][j])
                uStar[i][j] = uStar[i][j] + uCorr[i][j]
                
        for i in range(1, Nx + 1):
            for j in range(1, Ny): 
                vCorr[i][j] = - d / aPv[i][j] * (pCorr[i][j + 1] - pCorr[i][j])
                vStar[i][j] = vStar[i][j] + vCorr[i][j]
        
# Toleransa ulasildi mi?

#        if (m <= M):
#            outerResNorm[m - 1] = massSource
#        outerRes[m - 1] = massSource / np.amax(outerResNorm) # Fluent
#
#        if (outerRes[m - 1] < epsP):
#            break       
    
    print("\nOuter iterations ended at m = %d and outerRes = %.6f" %(m, outerRes[m - 1]))
        
# Yakinsayan degerleri kaydet
    
    u = np.copy(uStar)
    v = np.copy(vStar)
    p = np.copy(pStar)    

# Ard-islemler: Dogrulama
     
#    cellCenterxy = np.linspace(0.0 - d / 2.0, L + d / 2.0, num=Nx + 2)
                   
    u_centerline = u[25, :]
#    v_centerline = v[:, Nx / 2]

    flu_Re_100_y,flu_Re_100_u = np.loadtxt ("C:/Users/User/Desktop/Re100",  unpack=True)
#    flu_Re_100_x,flu_Re_100_v = np.loadtxt ("C:/Users/utku/Dropbox/python/HAD_I_1617/v_Re_100",  unpack=True)
#    
#    plt.plot(flu_Re_100_y, flu_Re_100_u, cellCenterxy, u_centerline, 'o', flu_Re_100_x, 2*flu_Re_100_v, cellCenterxy, 2*v_centerline, 'o')
    plt.plot(flu_Re_100_u, flu_Re_100_y, u_centerline, cy)
    plt.draw()
    plt.pause(0.01)
    plt.clf()

elapsed_time = time.time() - start_time
print(elapsed_time)
