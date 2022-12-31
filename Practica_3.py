# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:03:52 2022

@author: adelu
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from scipy import signal
import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.mplot3d import Axes3D

#El odeint.
I1= 4
I2= 2
I3= 1

def angular(z,t):
    w1,w2,w3= z
    dzdt= -(I3-I2)*w2*w3/I1, -(I1-I3)*w1*w3/I2, -(I2-I1)*w1*w2/I3
    return dzdt

#Ejercicio 1.

#Condiciones iniciales.

z0= 5,1,1

t= np.linspace(0,5,1000)#Tiempo de simulación

sol= odeint(angular,z0,t)

omega1= sol[:,0]
omega2= sol[:,1]
omega3= sol[:,2]

plt.figure()

plt.plot(t,sol[:,0], label= ('$\omega_1$'))
plt.plot(t,sol[:,1], label= ('$\omega_2$'))
plt.plot(t,sol[:,2], label= ('$\omega_3$'))
plt.ylabel('w')
plt.xlabel('t')
plt.legend(loc= 'best')

plt.show()

#Voy a sacar los maximos.
picos1= signal.find_peaks(omega1)[0]#Esto nos da índices.
picos2= signal.find_peaks(omega2)[0]
picos3= signal.find_peaks(omega3)[0]

print(picos1)#Para que veas a qué merefiero con índices.

periodo1= t[picos1[1]]-t[picos1[0]]
periodo2= t[picos2[1]]-t[picos2[0]]
periodo3= t[picos3[1]]-t[picos3[0]]

print(periodo1)
print(periodo2)
print(periodo3)

#Ahora bien, la frecuencia angular es 2pi/T

frec1= 2*np.pi/periodo1
frec2= 2*np.pi/periodo2
frec3= 2*np.pi/periodo3

frec1anal= z0[0]*((I1-I2)*(I1-I3)/(I2*I3))**0.5
print(frec1anal)

frec3anal= z0[2]*((I3-I2)*(I3-I1)/(I2*I1))**0.5#Esta es para el eje 3
print(frec3anal)

frec21anal= z0[1]*((I1-I2)*(I1-I3)/(I2*I3))**0.5#Esta es para el ejercicio 3
print(frec21anal)

frec23anal= z0[1]*((I3-I2)*(I3-I1)/(I2*I1))**0.5
print(frec23anal)

print(frec1)
print(frec2)
print(frec3)
#En el caso del eje estable 1 las frecuencias que coinciden con la fórmula 
#son las de las oscilaciones 2 y 3, si varias las condiciones para el caso
#estable 3, serán las de 1 y 2 :D

#Ejercicio 2.

#Las expresiones de la Ec y el momento angular son las siguientes...
#Ec= (I1w1^2 + I2w2^2 + I3w3^3)/2
#L^2= I1^2w1^2 + I2^2w2^2 + I3^2w3^2

Ec= 0.5*(I1*omega1**2 + I2*omega2**2 + I3*omega3**2)
L= ((I1*omega1)**2 + (I2*omega2)**2 + (I3*omega3)**2)**(1/2)

plt.figure()

plt.plot(t,Ec, label= '$Energía \ cinética$')
plt.plot(t,L, label= '$Momento \ angular$')
plt.legend(loc= 'best')
plt.xlabel('t')

plt.show()

#Ejercicio 3. Hacer lo del 1 pero para el eje intermedio.

#Ejercicio 4. Hacer lo mismo del 2 pero al eje intermedio.

#Ejercicio 5.

#Graficamos w
plt.figure() 

plt.subplot(111, projection='3d')
plt.plot(omega1,omega2,omega3)
plt.xlabel('$\omega_1$')
plt.ylabel('$\omega_2$')

plt.show()

#Graficamos L
plt.figure() 

plt.subplot(111, projection='3d')
plt.plot(I1*omega1,I2*omega2,I3*omega3, c= 'y')
plt.xlabel('$L_1$')
plt.ylabel('$L_2$')

plt.show()

#Variamos el tiempo de integración para sacar los periodos.

#Ejercicio 6.
