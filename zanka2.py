import os.path, sys
sys.path.append(os.path.dirname("zanka2.py"))

import time as ti
import math as ma
import numpy as np
import scipy as sps
import matplotlib.pyplot as plt
from scipy.integrate import quad
import zankapi as zp
import re


def norma2(r):
    return (r[0]**2 + r[1]**2 + r[2]**2)

def meje(dz, a):
    mej = [ [0, a], [-a, a], [0, a], [0, a], [-a, a], [0, a] ]
    return mej[dz]

def H(r, fi, a=1.0, I=1.0):
    """Input: r - (x, y, z), fi, a, I ((((aaarrrgggghhh))))
    """
    hx = 0.0
    hy = 0.0
    hz = 0.0
    args = ([a, fi, r])
    for i in range(1, 7):
        x1, x2 = meje(i-1, a)
        for s in ("x", "y", "z"):
            #funkcija = globals()["podint"+str(i)+s]
            funkcija = getattr(zp, "podint"+str(i)+s) # tako pa v modulu zp
            #print funkcija
            integral = quad(funkcija, x1, x2, args)
            if s=="x":
                hx += integral[0]
                #print "hx", i, hx
            elif s=="y":
                hy += integral[0]
                #print "\thy", i, hy
            elif s=="z":
                hz += integral[0]
                #print "\t\thz", i, hz
    h = np.array([hx, hy, hz])
    #h = h*I/(4*ma.pi)
    return h

def Htest(r, fi, a=1.0, I=1.0):
    hxs, hys, hzs, hxsa, hysa, hzsa = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    args = ([a, fi, r])
    for i in range(1, 7):
        x1, x2 = meje(i-1, a)
        for s in ("x", "y", "z"):
            #funkcija = globals()["podint"+str(i)+s] # tako dobim funkcijo iz stringa v tem modulu
            funkcija = getattr(zp, "podint"+str(i)+s) # tako pa v modulu zp
            funkcijaa = getattr(zp, "podint"+str(i)+s+"a")
            #print funkcija, funkcijaa
            integral = quad(funkcija, x1, x2, args)
            integrala = quad(funkcijaa, x1, x2, args)
            if s=="x":
                hx, hxa = integral[0], integrala[0]
                hxs = np.append(hxs, np.array([hx]))
                hxsa = np.append(hxsa, np.array([hxa]))
                #print "hx", i, hx, hxa
            elif s=="y":
                hy, hya = integral[0], integrala[0]
                hys = np.append(hys, np.array([hy]))
                hysa = np.append(hysa, np.array([hya]))
                #print "\thy", i, hy, hya
            elif s=="z":
                hz, hza = integral[0], integrala[0]
                hzs = np.append(hzs, np.array([hz]))
                hzsa = np.append(hzsa, np.array([hza]))
                #print "\t\thz", i, hz, hza
            else:
                print "Something went terribly wrong"
                return 0
    #h = np.array([hx, hy, hz])
    #h = h*I/(4*ma.pi)
    #print "hxs\thys\thzs\thxsa\thysa\thzsa"
    #for i in range(6):
    return hxs, hys, hzs, hxsa, hysa, hzsa

    

def tokovnica(rzac, fi, a=1.0, I=1.0, odd = 10.0, dr = 0.02):
    """Kaj hocem. vnesem zacetno tocko in potem ji sledim, dokler ni oddaljena odd*a. Sledim ji najprej v smeri H, potem pa se v nasprotni smeri...
    Input: rzac: zacetna tocka, ostalo je jasno
    Output: file oblike #r1x   r1y   r1z   norma(r1)   H1x   H1y   H1z   norma(H1)     # in potem tako naprej
    """
    # Koliko tock hocem - toliko da bo pregledno... recimo 500 Kako bi naredil, da bo okrog 500 tock? 10/500 = 0.02
    # Lahko naredim tako, da se vedno premaknem za doloceno vrednost, samo da potem z barvo oznacim velikost 
    # Ta ideja mi ni vsec, ker se mi zdi, da to ni ista silnica.. dodajam normo... Valda da je:) sam za 0.2 se 
    # premaknem  smeri H
    ime = "silnica" + re.sub(',', '', re.sub(' +', '_', re.sub('\.', '', re.sub('[\[\]]', '', str(rzac))))) + '.txt'
    f = open(ime, 'w')
    f.write("# rx\try\trz\tnorma(r)\tHx\tHy\tHz\tnorma(H)\n")
    hzac = H(rzac, fi, a, I)
    f.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'%(rzac[0], rzac[1], rzac[2], ma.sqrt(norma2(rzac)), hzac[0], hzac[1], hzac[2], ma.sqrt(norma2(hzac))))
    rnov = rzac + hzac/ma.sqrt(norma2(hzac))*dr
    for i in range(500):
        if ma.sqrt(norma2(rnov)) > a*odd:
            print "breakam 1"
            break
        h = H(rnov, fi, a, I)
        f.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'%(rnov[0], rnov[1], rnov[2], ma.sqrt(norma2(rnov)), h[0], h[1], h[2], ma.sqrt(norma2(h))))
        rnov = rnov + h/ma.sqrt(norma2(h))*dr
    
    f.write("\n\n")
    rnov = rzac - hzac/ma.sqrt(norma2(hzac))*dr
    for i in range(500):
        if ma.sqrt(norma2(rnov)) > a*odd:
            print "breakam 2"
            break
        h = H(rnov, fi, a, I)
        f.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n'%(rnov[0], rnov[1], rnov[2], ma.sqrt(norma2(rnov)), h[0], h[1], h[2], ma.sqrt(norma2(h))))
        rnov = rnov - h/ma.sqrt(norma2(h))*dr
    # Bomo videli, ce pridemo okrog:) in kolikokrat
    f.close()
    return ime

def silnica(fi, rzac, sttock = 500, a=1.0, I=1.0, odd = 10.0, drzac = 0.02, eps = 10**(-3)): # Silnica s prilagodljivim korakom
    """Input: fi - kot, rzac - zac tocka
    Output: RR - tocka na silnici;  HH[i] - size of magnetic field at RR[i]
    """
    RR = np.array([rzac])
    r1 = rzac
    rr = rzac
    hzac = H(rzac, fi, a, I)
    h1 = hzac
    HH = np.array([np.append(hzac, ma.sqrt(norma2(hzac)) )])
    raz, st, dr = 10.0, 0, drzac
    while raz>eps:
        r1 = rr + dr*h1#/ma.sqrt(norma2(h1))
        rm = rr + dr*h1/2.0#/(2.0*ma.sqrt(norma2(h1)))
        h2 = H(rm, fi, a, I)
        r2 = rm + dr*h2/2.0#/(2.0*ma.sqrt(norma2(h2)))
        raz = ma.sqrt(norma2(r1-r2))
        dr = dr/2.0
        st += 1
    RR = np.concatenate((RR, [r2]), axis=0)
    HH = np.concatenate((HH, [np.append(h2, ma.sqrt(norma2(hzac)))]), axis=0)
    dr, rr = drzac, r2
    for i in range(sttock/2 - 1):
        raz, st, = 10.0, 0
        while raz>eps:
            r1 = rr + dr*h1#/ma.sqrt(norma2(h1))
            rm = rr + dr*h1/2.0#/(2.0*ma.sqrt(norma2(h1)))
            h2 = H(rm, fi, a, I)
            r2 = rm + dr*h2/2.0#/(2.0*ma.sqrt(norma2(h2)))
            raz = ma.sqrt(norma2(r1-r2))
            dr = dr/2.0
            st += 1
        dr, rr = dr*4.0, r2
        r1, h1 = r2, H(r2, fi, a, I)
        RR = np.concatenate((RR, [r2]), axis=0)
        HH = np.concatenate((HH, [np.append(h2, ma.sqrt(norma2(hzac)))]), axis=0)
        #print i, st, dr, r2
    r1, h1, dr = rzac, hzac, drzac
    for i in range(sttock/2-1):
        raz, st  = 10.0, 0
        while raz>eps:
            r1 = r1 - dr*h1#/ma.sqrt(norma2(h1))
            rm = r1 - dr*h1/2.0#/(2.0*ma.sqrt(norma2(h1)))
            h2 = H(rm, fi, a, I)
            r2 = rm - dr*h2/2.0#/(2.0*ma.sqrt(norma2(h2)))
            raz = ma.sqrt(norma2(r1-r2))
            dr = dr/2.0
            st += 1
        dr, rr = dr*4.0, r2
        r1, h1 = r2, H(r2, fi, a, I)
        RR = np.concatenate(([r2], RR), axis=0)
        HH = np.concatenate((HH, [np.append(h2, ma.sqrt(norma2(hzac)))]), axis=0)
    return RR, HH

def tokovnica_cela(fi, zactocke, sttock = 1000, a=1.0, I=1.0, odd=10.0, dr=0.2, eps = 10**(-6)): # Popravi, da se bodo vsi argumenti funkcij ujemali! sttock s sttock v funkciji silnica
    """
    Naredi velik tokovnic in jih zapise v fajl...
    Input: fi - kot, tocke - zacetne tocke jih enakomerno razporedi po xy ravninini, odd - do katere oddaljenosti gledamo, dr - premik"""
    # Fino bi bilo, ce bi bil dr prilagodljiv, da ne pobegnem iz silnice dol...
    stzactock = len(zactocke) # ( to je ubistvu np.array tock [r1, r2, r3, ...] )
    #print fi
    ime = re.match('^\d\d?', re.sub('\.', '', str(fi))).group(0)
    ime = "grafi1/vse_" + ime + '_' + str(stzactock) + '.txt' # ime = kot # A je tukaj kej pomembno st tock je
    f = open(ime, 'w')
    f.write("# veliko tokovnic po stolpcih za fi = %.4f, a = %.4f, I = %.4f\n"%(fi, a, I))
    R = np.zeros((sttock, 3*stzactock))
    st = 0
    for rzac in zactocke:
        rr, hh = silnica(fi, rzac, sttock, a, I, odd, dr, eps) # Moram vse skupaj v R zapisat...
        #print rr.shape, R[:, st:(st+3)].shape # Tezava... odvisno a je liho ali sodo stttock
        R[:, st:(st+3)] = rr
        st += 3
        print st/3, "\t", rzac
    for i in range(sttock):
        for j in range(0, stzactock*3, 3):
            f.write("%.6f\t%.6f\t%.6f\t"%(R[i, j], R[i, j+1], R[i, j+2]))
        f.write("\n")
    f.close()
    return ime

def narisi_zanko(fi, sttock=10, a=1.0, write=True):
    """
    input: fi - kot, sttock - najbolje kar 10, a = 1.0, write (output)
    output: if write: datoteko za v gnuplot
            else:     vrne array tock na zanki...
    """
    ime = re.match('^\d\d?', re.sub('\.', '', str(fi))).group(0)
    ime = "grafi1/zanka_"+ ime + '.txt'
    f = open(ime, 'w')
    f.write("# tocke na zanki\n")
    rzac = np.array([0.0, -a, 0.0])
    RR = np.array([rzac])
    ii, jj , kk= np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([ma.cos(fi), 0, ma.sin(fi)])
    dr = a*1.0/(sttock-2)
    for i in range(sttock-2):
        rnov = RR[-1] + ii*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[a, -a, 0.0]]), axis=0)
    for i in range(2*sttock-4):
        rnov = RR[-1] + jj*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[a, a, 0.0]]), axis=0)
    for i in range(sttock-2):
        rnov = RR[-1] - ii*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[0.0, a, 0.0]]), axis=0)
    for i in range(sttock-2):
        rnov = RR[-1] + kk*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[a*ma.cos(fi), a, a*ma.sin(fi)]]), axis=0)
    for i in range(2*sttock-4):
        rnov = RR[-1] - jj*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[a*ma.cos(fi), -a, a*ma.sin(fi)]]), axis=0)
    for i in range(sttock-2):
        rnov = RR[-1] - kk*dr
        RR = np.concatenate((RR, [rnov]), axis=0)
    RR = np.concatenate((RR, [[0.0, -a, 0.0]]), axis=0)
    if write == True:
        for i in range(len(RR)):
            f.write("%.4f\t%.4f\t%.4f\n"%(RR[i, 0], RR[i, 1], RR[i, 2]))
        f.close()
        return ime
    else:
        return RR
    
            
zactocke2 = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.3, 0.0], [0.3, 0.6, 0.0], [0.3, 0.9, 0.0], [0.6, 0.0, 0.0], [0.6, 0.3, 0.0], [0.6, 0.6, 0.0], [0.6, 0.9, 0.0], [0.9, 0.0, 0.0], [0.9, 0.3, 0.0], [0.9, 0.6, 0.0], [0.9, 0.9, 0.0], [0.003, 0.0, 0.0], [0.003, 0.003, 0.0], [0.0, 0.003, 0.0], [0.0, 0.3, 0.0], [0.6, 0.3, 0.0], [0.9, 0.3, 0.0], [0.0, 0.6, 0.0], [0.3, 0.6, 0.0], [0.9, 0.6, 0.0], [0.0, 0.9, 0.0], [0.3, 0.9, 0.0], [0.6, 0.9, 0.0] ])

def zactocke(M=10, N=10, a=1.0):
    """
    M: st zac tock v s smeri
    N: st zac tock v y"""
    MM = np.delete(np.linspace(-a, a, M+2), [0, M+1])
    NN = np.delete(np.linspace(0, a, N+2), [0, N+1])
    ZAC = np.zeros((M*N+1, 3))
    for i in range(M*N):
        ZAC[i, 0] = MM[i/M]
        ZAC[i, 1] = NN[i%N]
    return ZAC

def cela(fi, zacM = 3, zacN = 3, sttock = 1000, a = 1.0, I=1.0, odd = 10.0, dr = 0.02, eps = 10**(-3)):
    """
    Vrne datoteko z vsemi slinicami, datoteko z zanko in gnuplot skripto, ki to narise
    """
    zacetne = zactocke(zacM, zacN, a)
    ime_t = tokovnica_cela(fi, zacetne, sttock, a, I, odd, dr, eps)
    print ime_t
    ime_z = narisi_zanko(fi, zacM, a, True)
    print ime_z
    ime_g = gpskripta(fi, len(zacetne))
#    ime_g = re.sub('\.txt', '.p', ime_t)
#    f = open(ime_g, 'w')
#    f.write("a = \"%s\" \n" %re.sub('grafi/', '', ime_t))
#    f.write("b = \"%s\"" %re.sub('grafi/', '', ime_z))
#    for i in range(len(zacetne)):
#        f.write("a using %d:%d:%d wi li linecolor rgb \"blue\",  " %(3*i, 3*i+1, 3*i+2))
    return "Uf"
        
def celacela():
    FI = [0.000, ma.pi/10, ma.pi/5, 3*ma.pi/10, 2*ma.pi/5, ma.pi/2, 3*ma.pi/5, 7*ma.pi/10, 4*ma.pi/5, 9*ma.pi/10, ma.pi]
    #FI = [ma.pi/5, 3*ma.pi/10, 2*ma.pi/5, ma.pi/2, 3*ma.pi/5, 7*ma.pi/10, 4*ma.pi/5, 9*ma.pi/10, ma.pi]
    #FI = [0.000, ma.pi/10, ma.pi/5]#, 3*ma.pi/10, 2*ma.pi/5, ma.pi/2, 3*ma.pi/5, 7*ma.pi/10, 4*ma.pi/5, 9*ma.pi/10, ma.pi]
    for fi in FI:
        cela(fi)# tko smo testali..., 6, 6, 10)
        print fi
    return None
    
def gpskripta(fi, stzactock = 100):
    # ime_g - ime gnuplot skripte (vse...)
    # ime_t - ime grafa, ime_z - ime zanke
    #print str(fi)
    ime_t = re.match('^\d\d?', re.sub('\.', '', str(fi))).group(0)
    ime_t = "grafi1/vse_" + ime_t + '_' + str(stzactock) + '.txt' # ime = kot # A je tukaj kej pomembno st tock je
    ime_z = re.match('^\d\d?', re.sub('\.', '', str(fi))).group(0)
    ime_z = "grafi1/zanka_"+ ime_z + '.txt'
    ime_g = re.sub('\.txt', '.p', ime_t)
    ime_t = re.sub('vse', 'Z', ime_t)
    f = open(ime_g, 'w')
    print ime_t, ime_g,ime_z, fi
   
    f.write("set term postscript enhance color\nset output \"%s\" \n\n"%(re.sub('^grafi1/', '', re.sub('\.txt', '.eps', ime_t))))
    f.write("unset key\n")
    f.write("set grid\n\n")
    f.write("set xrange [-3:3]\n")
    f.write("set yrange [-3:3]\n")
    f.write("set zrange [-3:3]\n\n")
    f.write("a = \"%s\" \n" %re.sub('grafi1/', '', ime_t))
    f.write("z = \"%s\" \n\n" %re.sub('grafi1/', '', ime_z))
    f.write("splot z using %d:%d:%d wi li lw 3 linecolor rgb \"black\" " %(1, 2, 3))
    for i in range(stzactock):
        f.write(", ")
        f.write(" a using %d:%d:%d wi li linecolor rgb \"blue\"" %(3*i+1, 3*i+2, 3*i+3))
        #f.write(" a every 10 using %d:%d:%d wi li" %(3*i+1, 3*i+2, 3*i+3)) # pisano:)
    return ime_g

def test():
    print "neki"
    pass
