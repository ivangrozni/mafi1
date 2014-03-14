import time as ti
import math as ma
import numpy as np

##############################################################################################
##############################################################################################
################################## FI = poljuben #############################################
##############################################################################################
##############################################################################################


#################### DEL ZANKE 1: ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5) ####################
def podint1x(l, param):
    """
    Input: param0 - a, param1 - fi, param2 - r
    """
    return 0
def podint1y(l, param):
    a = param[0]
    x, y, z = param[2]
    return -z/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)
def podint1z(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (y+a)/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)

#################### DEL ZANKE 2: ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5) ####################
def podint2x(l, param):
    a = param[0]
    x, y, z = param[2] 
    return z/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)
def podint2y(l, param):
    return 0.0
def podint2z(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (-x + a)/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)

#################### DEL ZANKE 3: ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5) ####################
def podint3x(l, param):
    return 0
def podint3y(l, param):
    a = param[0]
    x, y, z = param[2]
    return z/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)
def podint3z(l, param):
    a = param[0]
    x, y, z = param[2]
    return (-y+a)/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)

#################### DEL ZANKE 4: ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 1.5) ####################
def podint4x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return (-y+a)*ma.sin(fi)/ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 1.5)
def podint4y(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ((x-l*ma.cos(fi))*ma.sin(fi) - (z-l*ma.sin(fi))*ma.cos(fi) )/ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 1.5)
def podint4z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ((y-a)*ma.cos(fi) )/ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 1.5)

#################### DEL ZANKE 5: ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 1.5) ####################
def podint5x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return (-z + a*ma.sin(fi) )/ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 1.5)
def podint5y(l, param):
    return 0.0 #(-z + a*ma.sin(fi) )/ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 1.5)
def podint5z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return (x - a*ma.cos(fi) )/ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 1.5)

#################### DEL ZANKE 6: ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 1.5) ####################
def podint6x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ( (y+a)*ma.sin(fi) )/ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 1.5)
def podint6y(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ( (z-l*ma.sin(fi))*ma.cos(fi) - (x-l*ma.cos(fi))*ma.sin(fi) )/ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 1.5)
def podint6z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ( -(y+a)*ma.cos(fi) )/ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 1.5)

##############################################################################################
##############################################################################################
################################## FI = PI ###################################################
##############################################################################################
##############################################################################################


#################### DEL ZANKE 1: ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5) ####################
def podint1xa(l, param):
    """
    Input: param0 - a, param1 - fi, param2 - r
    """
    return 0
def podint1ya(l, param):
    a = param[0]
    x, y, z = param[2]
    return -z/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)
def podint1za(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (y+a)/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)

#################### DEL ZANKE 2: ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5) ####################
def podint2xa(l, param):
    a = param[0]
    x, y, z = param[2] 
    return z/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)
def podint2ya(l, param):
    return 0.0
def podint2za(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (-x + a)/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)

#################### DEL ZANKE 3: ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5) ####################
def podint3xa(l, param):
    return 0
def podint3ya(l, param):
    a = param[0]
    x, y, z = param[2]
    return z/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)
def podint3za(l, param):
    a = param[0]
    x, y, z = param[2]
    return (-y+a)/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)

#################### DEL ZANKE 4: ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5) ####################
def podint4xa(l, param):
    a = param[0]
    x, y, z = param[2] 
    return 0.0
def podint4ya(l, param):
    a = param[0]
    x, y, z = param[2] 
    return z/ma.pow((x+l)**2 + (y-a)**2 + z**2, 1.5)
def podint4za(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (-y+a)/ma.pow((x+l)**2 + (y-a)**2 + z**2, 1.5) ###!!!

#################### DEL ZANKE 5: ma.pow((x+a)**2 + (y-l)**2 + z**2, 1.5) ####################
def podint5xa(l, param):
    a = param[0]
    x, y, z = param[2] 
    return -z/ma.pow((x+a)**2 + (y-l)**2 + z**2, 1.5)
def podint5ya(l, param):
    return 0.0
def podint5za(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (x + a)/ma.pow((x+a)**2 + (y-l)**2 + z**2, 1.5)

#################### DEL ZANKE 6: ma.pow((x+l)**2 + (y+a)**2 + z**2, 1.5) ####################
def podint6xa(l, param):
    return 0
def podint6ya(l, param):
    a = param[0]
    x, y, z = param[2]
    return -z/ma.pow((x+l)**2 + (y+a)**2 + z**2, 1.5)
def podint6za(l, param):
    a = param[0]
    x, y, z = param[2]
    return (y+a)/ma.pow((x+l)**2 + (y+a)**2 + z**2, 1.5)

##############################################################################################
##############################################################################################
################################## FI = PI/2 #################################################
##############################################################################################
##############################################################################################


#################### DEL ZANKE 1: ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5) ####################
def podint1xb(l, param):
    """
    Input: param0 - a, param1 - fi, param2 - r
    """
    return 0
def podint1yb(l, param):
    a = param[0]
    x, y, z = param[2]
    return -z/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)
def podint1zb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (y+a)/ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5)

#################### DEL ZANKE 2: ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5) ####################
def podint2xb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return z/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)
def podint2yb(l, param):
    return 0.0
def podint2zb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return (-x + a)/ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5)

#################### DEL ZANKE 3: ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5) ####################
def podint3xb(l, param):
    return 0
def podint3yb(l, param):
    a = param[0]
    x, y, z = param[2]
    return z/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)
def podint3zb(l, param):
    a = param[0]
    x, y, z = param[2]
    return (-y+a)/ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5)

#################### DEL ZANKE 4: ma.pow(x**2 + (y-a)**2 + (z-l)**2, 1.5) ####################
def podint4xb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return -(y-a)/ma.pow(x**2 + (y-a)**2 + (z-l)**2, 1.5)
def podint4yb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return x/ma.pow(x**2 + (y-a)**2 + (z-l)**2, 1.5)  
def podint4zb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return 0.0

#################### DEL ZANKE 5: ma.pow(x**2 + (y-l)**2 + (z-a)**2, 1.5) ####################
def podint5xb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return -(z-a)/ma.pow(x**2 + (y-l)**2 + (z-a)**2, 1.5)
def podint5yb(l, param):
    return 0.0
def podint5zb(l, param):
    a = param[0]
    x, y, z = param[2] 
    return x/ma.pow(x**2 + (y-l)**2 + (z-a)**2, 1.5)

#################### DEL ZANKE 6: ma.pow(x**2 + (y+a)**2 + (z-l)**2, 1.5) ####################
def podint6xb(l, param):
    a = param[0]
    x, y, z = param[2]
    return -(y+a)/ma.pow(x**2 + (y+a)**2 + (z-l)**2, 1.5)
def podint6yb(l, param):
    a = param[0]
    x, y, z = param[2]
    return -x/ma.pow(x**2 + (y+a)**2 + (z-l)**2, 1.5)#z/ma.pow((x+l)**2 + (y+a)**2 + z**2, 1.5)
def podint6zb(l, param):
    a = param[0]
    x, y, z = param[2]
    return 0.0#-(y+a)/ma.pow((x+l)**2 + (y+a)**2 + z**2, 1.5)



##############################################################################################
##############################################################################################
################################## FI = poljuben #############################################
############################### Vektorski potencial ##########################################
##############################################################################################
##############################################################################################


#################### DEL ZANKE 1: ma.pow((x-l)**2 + (y+a)**2 + z**2, 1.5) ####################
def apodint1x(l, param):
    """
    Input: param0 - a, param1 - fi, param2 - r
    """
    a = param[0]
    x, y, z = param[2]
    return 1.0/ma.pow((x-l)**2 + (y+a)**2 + z**2, 0.5)
def apodint1y(l, param):
    a = param[0]
    x, y, z = param[2]
    return 0.0
def apodint1z(l, param):
    a = param[0]
    x, y, z = param[2] 
    return 0.0

#################### DEL ZANKE 2: ma.pow((x-a)**2 + (y-l)**2 + z**2, 1.5) ####################
def apodint2x(l, param):
    a = param[0]
    x, y, z = param[2] 
    return 0.0
def apodint2y(l, param):
    a = param[0]
    x, y, z = param[2]
    return 1.0/ma.pow((x-a)**2 + (y-l)**2 + z**2, 0.5)
def apodint2z(l, param):
    a = param[0]
    x, y, z = param[2] 
    return 0.0

#################### DEL ZANKE 3: ma.pow((x-l)**2 + (y-a)**2 + z**2, 1.5) ####################
def apodint3x(l, param):
    a = param[0]
    x, y, z = param[2]
    return -1.0/ma.pow((x-l)**2 + (y-a)**2 + z**2, 0.5)
def apodint3y(l, param):
    a = param[0]
    x, y, z = param[2]
    return 0.0
def apodint3z(l, param):
    a = param[0]
    x, y, z = param[2]
    return 0.0

#################### DEL ZANKE 4: ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 1.5) ####################
def apodint4x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ma.cos(fi)/ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 0.5)
def apodint4y(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return 0.0
def apodint4z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return ma.sin(fi)/ma.pow((x-l*ma.cos(fi))**2 + (y-a)**2 + (z -l*ma.sin(fi))**2, 0.5)

#################### DEL ZANKE 5: ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 1.5) ####################
def apodint5x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return 0.0
def apodint5y(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return -1.0/ma.pow((x-a*ma.cos(fi))**2 + (y-l)**2 + (z -a*ma.sin(fi))**2, 0.5)
def apodint5z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return 0.0

#################### DEL ZANKE 6: ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 1.5) ####################
def apodint6x(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return -ma.cos(fi) /ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 0.5)
def apodint6y(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return 0.0
def apodint6z(l, param):
    a, fi = param[0], param[1]
    x, y, z = param[2]
    return -ma.sin(fi)/ma.pow((x-l*ma.cos(fi))**2 + (y+a)**2 + (z-l*ma.sin(fi))**2, 0.5)
