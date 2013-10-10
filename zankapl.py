import os.path, sys
sys.path.append(os.path.dirname("zankapl.py"))

import time as ti
import math as ma
import numpy as np
import scipy as sps
import matplotlib.pyplot as plt
#from scipy.integrate import quad
import zankapi as zp
import zanka2 as zd
#import re

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from matplotlib import cm # colormap

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#a = Arrow3D([0,1],[0,1],[0,1], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
#ax.add_artist(a)
#plt.show()

def kot(fi, a = 1.0, st_tock=100):
    """Vrne krivuljo po kotu """
    r = a/2.0
    RR = np.array([[a/2.0, a, 0.0]])
    FI = np.linspace(0.0, fi, st_tock)
    for i in range(1, st_tock):
        rnov = [a/2.0*ma.cos(FI[i]), a, a/2.0*ma.sin(FI[i])]
        RR = np.concatenate((RR, [rnov]), axis=0)
    return RR

def zanka(fi = 1.57, risi=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    RR = narisi_zanko(fi, 10, 1.0, False)
    
    ax.plot(RR[:, 0], RR[:, 1], RR[:, 2], color='red', linewidth=2.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if risi==False: return ax
    a = Arrow3D([1.1,1.1],[-0.5,0.5],[0,0], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    RR = kot(fi, 1.0, 100)
    ax.plot(RR[:, 0], RR[:, 1], RR[:, 2], color='black', linewidth=0.8)
    
    ax.text(0.2*ma.cos(fi/2), 1.0, 0.2*ma.sin(fi/2.0), r'$\alpha$', color='black')
    
    ax.text(1.22, 0.0, 0.0, 'tok', color='black')

    ax.text(0.5, -0.8, 0.0, 'I', color='black')
    ax.text(0.8, 0.0, 0.0, 'II', color='black')
    ax.text(0.5, 0.8, 0.0, 'III', color='black')
    ax.text(ma.cos(fi)/2.0, 0.8, ma.sin(fi)/2.0, 'IV', color='black')
    ax.text(ma.cos(fi)-0.2, 0.0, ma.sin(fi)-0.2, 'V', color='black')
    ax.text(ma.cos(fi)/2.0, -0.8, ma.sin(fi)/2.0, 'VI', color='black')

    ax.set_zlim3d(-1.2, 1.2)         
    ax.set_ylim3d(-1.2, 1.2)
    ax.set_xlim3d(-1.2, 1.2)

    plt.show()
    return None

def podatki(fi, a=1.0, I=1.0):
    """Vrne podatke za narisat lep graf - torej precej tock v prostoru, in smeri kam kaze H v teh tockah in njegovo velikost.
    """
    t = ti.time()
    RR = np.array([[0.0, 0.0, 0.0]])
    HH = np.array([[0.0, 0.0, 0.0, 0.0]])
    for z in np.linspace(-1.5, 1.5, 9):
        print z
        for x in np.linspace(-1.5, 1.5, 9):
            for y in np.linspace(0, 1.5, 5):
                RR = np.concatenate((RR, [[x, y, z]]), axis=0)
                h = zd.H([x, y, z], fi, a, I)
                hn = ma.sqrt(norma2(h))
                HH = np.concatenate((HH, [ np.append(1.0*h/hn, hn) ] ), axis=0)
    RR = np.delete(RR, 0, 0)
    HH = np.delete(HH, 0, 0)
    tt = ti.time() - t
    print tt, 'operacij:%d\t'%(31*31*16), tt/(31*31*16)
    return RR, HH

def graf1(fi, RR, HH):
    """Narise vektorcke v prostoru... in jih lepo pobarva
    """
    n = len(RR)
    #ax = zanka(fi, False) # risem zanko
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = narisi_zanko(fi, 10, 1.0, False)
    ax.plot(z[:, 0], z[:, 1], z[:, 2], color='black', linewidth=2.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    h = HH[:, 3]
    hmax = max(h)
    hmin = min(h)
    for i in range(n):
        a = Arrow3D([RR[i, 0], RR[i, 0]+0.4*HH[i, 0]],[RR[i, 1],RR[i, 1]+0.4*HH[i, 1] ],[RR[i, 2], RR[i, 2]+0.4*HH[i, 2]], mutation_scale=20, lw=1, arrowstyle="-|>", color=((h[i] - hmin)/(hmax-hmin), 0, 1 - (h[i]-hmin)/(hmax-hmin)  ) ) # RGB color
        ax.add_artist(a)
    
    #ax.title(r'$\alpha=$'+'%.3f'%fi)
    ax.set_zlim3d(-1.5, 1.5)         
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_xlim3d(-1.5, 1.5)

    plt.show()
    return None

##################################################################
############################## z os ##############################
##################################################################

def pdatki2(st_tock, st_kotov=100):
    """Narise graf velikosti magnetnega polja na z osi v odvisnosti od kota
    """
    # OK to je mogoce malce neumno. Fino bi bilo imeti mag polje po sredini zanke...
    # Oboje bom pogledal
    Z = np.linspace(-3, 3, st_tock)
    FI = np.linspace(-2*pi, 2*pi, st_kotov)
    HH = np.array([0, 0, 0])
    REZ = [np.zeros(st_kotov)]
    for i in range(st_tock):
        rez = []
        for j in range(st_kotov):
            h = zd.H([0, 0, Z[i]], FI[j] )
            hn = ma.sqrt(norma2(h))
            rez = np.append(rez, hn)
        REZ = np.concatenate((REZ, [[rez]]), axis=0)
    REZ = np.delete(REZ, 0, 0)
    
    return Z, FI, REZ

def generate(z, fi, a=1.0, I=1.0):
    h = zd.H([0.0, 0.0, z], fi, a, I)
    hn = ma.sqrt(zd.norma2(h))
    if hn > 20:
        return 20
    return hn

def generate2(l, fi, a=1.0, I=1.0):
    x = l*ma.cos(fi/2.0)
    z = l*ma.sin(fi/2.0)
    h = zd.H([x, 0.0, z], fi, a, I)
    hn = ma.sqrt(zd.norma2(h))
    if hn > 20:
        return 20
    return hn

def graf2(ztock = 50, fitock=50):
    plt.ion()
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')

    zs = np.linspace(-4, 4, ztock)
    fs = np.linspace(0, 2*ma.pi, fitock)
    ZS, FS = np.meshgrid(zs, fs)
    
    # Imel bi se tabelo zs-jev, pri katerih je max, pri dolocenem fi

    REZ = [np.zeros(fitock)]
#    f = open("rez2.txt", 'w')
#    f.write("# velikost H v odvisnosti od kota in razdalje od izhodisca na silnici, ki gre skozi izhodisce\n")
#    f.write("# L\tFI\thabs\n")

    for i in range(ztock):
        H = []
        for j in range(fitock):
            h = generate2(ZS[i, j], FS[i, j])
            H = np.append(H, h)
            #H = np.append(H, ma.log10(h))
            #f.write("%.4f\t%.4f\t%.4f\n"%(ZS[j, i], FS[j, i], h))
        REZ = np.concatenate((REZ, [H]), axis=0)
    REZ = np.delete(REZ, 0, 0)
    #surf = ax.plot_surface(ZS, FS, np.log(REZ),  rstride=1, cstride=1, linewidth=0, antialiased=False, cmap = cm.spectral)

    REZ = np.log10(REZ)
    (I, J) = REZ.shape
    for i in range(I):
        for j in range(J):
            if REZ[i, j] < -2: REZ[i, j] = -2.0

    cset = ax.contourf(ZS, FS, REZ, zdir='z', offset=-2.1, cmap=cm.coolwarm)
    cset = ax.contourf(ZS, FS, REZ, zdir='x', offset=-4.1, cmap=cm.coolwarm)
    #cset = ax.contourf(ZS, FS, REZ, zdir='y', offset=3.2, cmap=cm.coolwarm)
    cset = ax.contourf(ZS, FS, REZ, zdir='y', offset=6.3, cmap=cm.coolwarm)

    ax.set_xlabel('L')
    ax.set_ylabel('FI')
    ax.set_zlabel('log10(abs(H))')
    #ax.set_zlabel('abs(H)')

    surf = ax.plot_wireframe(ZS, FS, REZ, rstride=1, cstride=1, alpha=0.3, linewidth=0.2, color="black")#, cmap=cm.coolwarm)
    #ax.set_zlim(-2, 2)
    #fig.colorbar(cset, shrink=1, aspect=10)

    plt.show()
    #f.close()
    return REZ

def graf21(FI = [ma.pi/3, ma.pi/2, 2*ma.pi/3, ma.pi], L = np.linspace(-5, 5, 100)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for fi in FI:
        HH = []
        for l in L:
            HH = np.append(HH, generate2(l, fi))
        ax.plot(L, HH, 'o', label=r'$\phi = $%.2f'%(fi))


    #L = np.linspace(0, 5, 100)
    #L = np.delete(L, 0)
    #HH = []
    #for l in L:
    #    HH = np.append(HH, 2.0/(ma.pi*ma.pow(l, 3)))

        #print len(L), len(H), '\n', L, H
    #ax.plot(L, HH, 'k', label='2/(pi l l l)')

    ax.grid()
    ax.legend()                      
    ax.set_ylim(0, 10) 
                  
    plt.show()
    return None

def graf_oddal(N=1000):
    FI = [ma.pi/4, ma.pi/2, 3*ma.pi/4, ma.pi]
    zs = np.linspace(-5, 15, N)
    R = np.zeros((1000, 4))

    for i in range(len(FI)):
        for t in range(len(zs)):
            h = zd.H([zs[t]*ma.cos(FI[i]/2), 0.0, zs[t]*ma.sin(FI[i]/2)], FI[i], 1.0, 1.0)
            hnl = ma.log(ma.sqrt(zd.norma2(h)))
            #hnl = ma.sqrt(zd.norma2(h))
            R[t, i] = hnl
            

    plt.figure()
    plt.plot(zs - ma.sqrt(2*(ma.cos(FI[0]) +1))/2, R[:, 0], label="%.3f"%FI[0])
    plt.plot(zs - ma.sqrt(2*(ma.cos(FI[1]) +1))/2, R[:, 1], label="%.3f"%FI[1])
    plt.plot(zs - ma.sqrt(2*(ma.cos(FI[2]) +1))/2, R[:, 2], label="%.3f"%FI[2])
    plt.plot(zs - ma.sqrt(2*(ma.cos(FI[3]) +1))/2, R[:, 3], label="%.3f"%FI[3])
    plt.legend()
    plt.title("Logaritem velikosti mag. polja na srednici")
    #plt.title("Velikost mag. polja na srednici")
    plt.grid()
    plt.xlabel("oddaljenost od izhodisca")
    plt.ylabel("norm(H)")
    plt.show()

    return None

def graf_max(ztock = 100, fitock=100):
    """Vrne oddaljenost od izhodisca za maximum mag polja v odvisnosti od kota pregiba."""
    zs = np.linspace(-1.5, 1.5, ztock)
    fs = np.linspace(0, 2*ma.pi, fitock)
    ZS, FS = np.meshgrid(zs, fs)
    
    REZ = [np.zeros(fitock)]
    ZMAX = []

    for i in range(ztock):
        H = []
        for j in range(fitock): # spreminjam oddaljenost od
            #print ZS[i, j], "kot",FS[i, j]
            h = generate2(ZS[i, j], FS[i, j])
            H = np.append(H, h)
        # itemindex = np.where(array==item)[0][0]
        # nonzero(array == item)[0][0]
        itemindex = np.where(H==max(H))[0][0]
        np.nonzero(H == max(H))[0][0]
        # print itemindex, zs[int(itemindex)]
        ZMAX = np.append(ZMAX, zs[int(itemindex)])

        REZ = np.concatenate((REZ, [H]), axis=0)
    REZ = np.delete(REZ, 0, 0) # Kaj je REZ: Matrika, ki ima po stolpcih mag polje
    
    ana = []
    for fi in fs:
        ana = np.append(ana, ma.sqrt(2*(ma.cos(fi)+1))/2)
    
    # Grem po stolpcih in poiscem index-e maximumov
    # Potem vrnem narisem tabelo max v odvisnosti od kota pregiba.
    # Ni kul, ker se mi spreminjata sproti obe stvari - oddaljenost in polje, al kaj
    

    

    return ZMAX, ana, zs



###############################################################################
############################## polje na kroznici ##############################
###############################################################################

def generate3(fi, alpha, d, a=1.0, I=1.0):
    """ Generira velikosti polja na kroznici z radijem d 
    alpha - kot pri katerem meirimo mag polje, fi - kot prepogiba zanke
    """
    x = d*ma.cos(alpha/2.0)
    z = d*ma.sin(alpha/2.0)
    h = zd.H([x, 0.0, z], fi, a, I)
    hn = ma.sqrt(zd.norma2(h))
    if hn > 20:
        return 20
    return hn

def graf3(d, ztock = 50, fitock=50):
    """Narise graf velikosti H v odvisnosti od kota pregiba na krogu v ravnini xz okrog zanke"""
    plt.ion()
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')

    aas = np.linspace(0, 2*ma.pi, ztock)
    fs = np.linspace(0, 2*ma.pi, fitock)
    AS, FS = np.meshgrid(aas, fs)
    
    REZ = [np.zeros(fitock)]
#    f = open("rez2.txt", 'w')
#    f.write("# velikost H v odvisnosti od kota in razdalje od izhodisca na silnici, ki gre skozi izhodisce\n")
#    f.write("# L\tFI\thabs\n")
    for i in range(ztock):
        H = []
        for j in range(fitock):
            h = generate3(FS[i, j], AS[i, j], d)
            H = np.append(H, h)
            #f.write("%.4f\t%.4f\t%.4f\n"%(ZS[j, i], FS[j, i], h))
        REZ = np.concatenate((REZ, [H]), axis=0)
    REZ = np.delete(REZ, 0, 0)
    #surf = ax.plot_surface(ZS, FS, np.log(REZ),  rstride=1, cstride=1, linewidth=0, antialiased=False, cmap = cm.spectral)
#    REZ = np.log10(REZ)
#    (I, J) = REZ.shape
#    for i in range(I):
#        for j in range(J):
#            if REZ[i, j] < -2: REZ[i, j] = -2.0

    cset = ax.contourf(AS, FS, REZ, zdir='z', offset=-0.01, cmap=cm.coolwarm)
    cset = ax.contourf(AS, FS, REZ, zdir='x', offset=-0.1, cmap=cm.coolwarm)
    #cset = ax.contourf(ZS, FS, REZ, zdir='y', offset=3.2, cmap=cm.coolwarm)
    cset = ax.contourf(AS, FS, REZ, zdir='y', offset=6.3, cmap=cm.coolwarm)

    ax.set_xlabel('ALPHA')
    ax.set_ylabel('FI')
    ax.set_zlabel('(abs(H)')

    surf = ax.plot_wireframe(AS, FS, REZ, rstride=1, cstride=1, alpha=0.3, linewidth=0.2, color="black")#, cmap=cm.coolwarm)
    #ax.set_zlim(-2, 2)
    #fig.colorbar(cset, shrink=1, aspect=10)
    plt.title("magnetno polje na kroznici v ravnini xz z radijem d=%.2f"%d)

    plt.show()
    #f.close()
    return REZ

def graf31(d, atock=50, fitock=50):
    """Narise kroznico in jo pobarva po skali!
    """
    pass




##################################################################
############################## TEST ##############################
##################################################################

def grafVTK(fi, a=1.0, I = 1.0):
    """Zapise v .vtk file - da bom lahko narisal lepe tokovnice.
    """
    ymin = -2.0
    xmin = zmin = -2.0
    xmax = ymax = zmax = 2.0
    dimx = dimz = 16 # prej je bilo 32
    dimy = 16
    X = np.linspace(xmin, xmax, dimx)
    Y = np.linspace(ymin, ymax, dimy)
    Z = np.linspace(zmin, zmax, dimz)
    HN = []
    f = open('vtkgrafi2/bbb_%d.vtk'%(fi*100), 'w')
    f.write('# vtk DataFile Version 3.8\n')
    f.write('Mag polje okrog kvadratne zanke prepognjene za %d\n'%(fi*100)) # Morda kaka lepsa oblika
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_GRID\nDIMENSIONS %d %d %d\nPOINTS %d float\n'%(dimx, dimy, dimz, dimx*dimy*dimz))
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                f.write('%.6f %.6f %.6f\n'%(X[i], Y[j], Z[k]))
    f.write('\nPOINT_DATA %d\nVECTORS MagPoljeNorm float\n'%(dimx*dimy*dimz))
    for i in range(dimx): # samo smer mag polja
        for j in range(dimy):
            for k in range(dimz):
                h = zd.H([X[i], Y[j], Z[k]], fi, a, I)
                hn = ma.sqrt(zd.norma2(h))
                HN = np.append(HN, hn) # Tukaj imam matriko polj
                f.write('%.6f %.6f %.6f\n'%(h[0]/hn, h[1]/hn, h[2]/hn))
    f.write('\n\nVECTORS MagPolje float\n')
    for i in range(dimx): # cel vektor mag polja
        for j in range(dimy):
            for k in range(dimz):
                h = zd.H([X[i], Y[j], Z[k]], fi, a, I)
                hn = ma.sqrt(zd.norma2(h))
                f.write('%.6f %.6f %.6f\n'%(h[0], h[1], h[2]))
        print i
    f.write('\nSCALARS Norma float\nLOOKUP_TABLE default\n')
    nmin, nmax = min(HN), max(HN)
    for i in range(len(HN)):
        f.write('%.6f\n'%((HN[i] - nmin)/(nmax - nmin*1.0)))
    f.write('\nSCALARS LogNorma float\nLOOKUP_TABLE default\n')
    nmin, nmax = min(np.log(HN)), max(np.log(HN))
    for i in range(len(HN)):
        f.write('%.6f\n'%((np.log(HN[i]) - nmin)/(nmax - nmin*1.0)))
    # Probam vse zapisat v isti file :)
    sttock = 50
    RR = zd.narisi_zanko(fi, sttock, a, False)
    z = open('vtkgrafi2/aaa_%d.vtk'%(fi*100), 'w')
    z.write('# vtk DataFile Version 3.8\n')
    z.write('Kvadratna zanka prepognjena za %d\n'%(fi*100)) # Morda kaka lepsa oblika
    z.write('ASCII\n') # dim je cudna spemenljivka
    z.write("\n")
    dim = len(RR)
    z.write('DATASET UNSTRUCTURED_GRID\nPOINTS %d float\n'%(dim))
    for i in range(len(RR)):
        z.write('%.6f %.6f %.6f\n'%(RR[i, 0], RR[i, 1], RR[i, 2]))
    z.write('\nPOINT_DATA %d\nSCALARS Zanka float\nLOOKUP_TABLE default\n'%dim)
    for i in range(len(RR)):
        z.write('%.6f\n'%(1.0) )

    z.close()
    f.close()
    return None
                
# Kaj bi rad imel: Vse zanke in vsa polja in norme v enem vtk fajlu

def total_grafVTK(fi, a=1.0, I = 1.0):
    """Zapise v .vtk file - da bom lahko narisal lepe tokovnice.
    Za vse kote zapise v en vtk file:) recimo, da samo normirane vektorje mag polja
    """
    ymin = -2.0
    xmin = zmin = -2.0
    xmax = ymax = zmax = 2.0
    dimx = dimz = 16 # prej je bilo 32
    dimy = 16
    X = np.linspace(xmin, xmax, dimx)
    Y = np.linspace(ymin, ymax, dimy)
    Z = np.linspace(zmin, zmax, dimz)
    HN = []
    f = open('vtkgrafi/xxx.vtk', 'w')
    f.write('# vtk DataFile Version 3.8\n')
    f.write('Mag polje okrog kvadratne zanke prepognjene za ... \n') # Morda kaka lepsa oblika
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_GRID\nDIMENSIONS %d %d %d\nPOINTS %d float\n'%(dimx, dimy, dimz, dimx*dimy*dimz))
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                f.write('%.6f %.6f %.6f\n'%(X[i], Y[j], Z[k]))
    for fi in FI:
        f.write('\nPOINT_DATA %d\nVECTORS MagPoljeNorm%d float\n'%(dimx*dimy*dimz, fi*100))
        for i in range(dimx):
            for j in range(dimy):
                for k in range(dimz):
                    h = zd.H([X[i], Y[j], Z[k]], fi, a, I)
                    hn = ma.sqrt(zd.norma2(h))
                    HN = np.append(HN, hn)
                    f.write('%.6f %.6f %.6f\n'%(h[0]/hn, h[1]/hn, h[2]/hn))
    #    f.write('\n\nVECTORS MagPolje%d float\n' %(fi*100))
    #    for i in range(dimx):
    #        for j in range(dimy):
    #            for k in range(dimz):
    #                h = zd.H([X[i], Y[j], Z[k]], fi, a, I)
    #                hn = ma.sqrt(zd.norma2(h))
    #                f.write('%.6f %.6f %.6f\n'%(h[0], h[1], h[2]))
    #        print i
        f.write('\nSCALARS Norma%d float\nLOOKUP_TABLE default\n' %fi*100)
        nmin, nmax = min(HN), max(HN)
        for i in range(len(HN)):
            f.write('%.6f\n'%((HN[i] - nmin)/(nmax - nmin*1.0)))
        f.write('\nSCALARS LogNorma%d float\nLOOKUP_TABLE default\n'%fi*100)
        nmin, nmax = min(np.log(HN)), max(np.log(HN))
        for i in range(len(HN)):
            f.write('%.6f\n'%((np.log(HN[i]) - nmin)/(nmax - nmin*1.0)))
    # Probam vse zapisat v isti file :)
    sttock = 50
    RR = zd.narisi_zanko(fi, sttock, a, False)
    z = open('vtkgrafi/zanxa.vtk', 'w')
    z.write('# vtk DataFile Version 3.8\n')
    z.write('Kvadratna zanka prepognjena za %d\n'%(fi*100)) # Morda kaka lepsa oblika
    z.write('ASCII\n') # dim je cudna spemenljivka
    z.write("\n")
    dim = len(RR)
    z.write('DATASET UNSTRUCTURED_GRID\nPOINTS %d float\n'%(dim)) # Ni kul
    for i in range(len(RR)):
        z.write('%.6f %.6f %.6f\n'%(RR[i, 0], RR[i, 1], RR[i, 2]))
    z.write('\nPOINT_DATA %d\n'%dim)
    for fi in FI:
        z.write('SCALARS Zanka%d float\nLOOKUP_TABLE default\n'%fi*100)
        for i in range(len(RR)):
            z.write('%.6f\n'%(1.0) )

    z.close()
    f.close()
    return None


"""
def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n)+ vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100

for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""




