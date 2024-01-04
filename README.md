111.   curve y=e**x

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-10,10,0.001)
y=np.exp(x)
plt.plot(x,y)
plt.title("exponential curve")
plt.grid()




222222.     sine cosine curves*****

x = np.arange(-10, 10, 0.001)
y1 = np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,x,y2)
plt.title("sine curve and cosine curve")
plt.xlabel("Values of x")
plt.ylabel("Values of sin(x) and cos(x) ")
plt.grid()
plt.show()


3333.    x**2+y**2=4

from sympy import plot_implicit, symbols, Eq
x,y=symbols('x,y')
p1= plot_implicit(Eq (x*2+y*2,4),(x,-4,4),(y,-4,4),title='CIRCLE')



4444444.      r=5(1+cos theta )

from pylab import *
theta=linspace (0,2*np.pi,1000)
r1=5+5*cos(theta)
polar (theta,r1,'r')
show()




5555.      angle betwen the curves r=4(1+cost) and r=5(1-cost)
from sympy import *
r,t =symbols('r,t')
r1=4*(1+cos(t));
r2=5*(1-cos(t));
dr1=diff(r1,t)
dr2=diff(r2,t)
t1=r1/dr1
t2=r2/dr2
q=solve(r1-r2,t)
w1=t1.subs({t:float(q[0])})
w2=t2.subs({t:float(q[0])})
y1=atan(w1)
y2=atan(w2)
w=abs(y1-y2)
print('Angle between curves in radians is %0.4f'%float(w))



6666.      radius of curvature of r=4(1+cost) at t=pi/2

