1.( A )  curve y=e**x

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-10,10,0.001)
y=np.exp(x)
plt.plot(x,y)
plt.title("exponential curve")
plt.grid()


1. ( B ) u=e**x(xcosy-ysiny)then Uxx+Uyy=0

from sympy import *
x,y=symbols('x,y')
u=exp(x)*(x*cos(y)-y*sin(y))
display(u)
dux=diff(u, x)
duy=diff(u, y)
uxx=diff(dux, x)
uyy=diff(duy,y)
w=uxx+uyy
w1=simplify(w)
print('Ans',float(w1))


2. ( A)  sine and cosine curve

x = np.arange(-10, 10, 0.001)
y1 = np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,x,y2)
plt.title("sine curve and cosine curve")
plt.xlabel("Values of x")
plt.ylabel("Values of sin(x) and cos(x) ")
plt.grid()
plt.show()


2.( B )    dy/dx +ytanx-y**3sec(x)=0

from sympy import *
x,y=symbols('x,y')
y=Function("y")(x)
y1=Derivative(y,x)
z1=dsolve (Eq (y1+y*tan(x)-y**3*sec(x),0),y)
display(z1)



3. ( A )     sine and cosine curve

x = np.arange(-10, 10, 0.001)
y1 = np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,x,y2)
plt.title("sine curve and cosine curve")
plt.xlabel("Values of x")
plt.ylabel("Values of sin(x) and cos(x) ")
plt.grid()
plt.show()



3.( B )     dy/dx+x**2y-x**5=0

from sympy import *
x,y=symbols('x,y')(x)
y1=Derivative(y,x)
z1=dsolve(Eq(y1+x**2*y-x**5,0),y)
display(z1)



5 .( A )  cardroid r=5(1+cos theta )

from pylab import *
theta=linspace(0,2*np.pi,1000)
r1=5+5*cos(theta)
polar(theta,r1,'r')
show()



5 . ( B )   beta (m,n)=gamma(m) gamma(n) / gamma(m+n)

from sympy import beta, gamma
m=5;
n=7;
m=float(m);
n=float(n);
s=beta(m,n);
t=(gamma(m)*gamma(n)/gamma(m+n));
print(s,t)
if(abs(s-t)<=0.00001):
    print('Beta and Gamma are related')
else:
    print('Given values are wrong')




6. ( A )   cicle x**2+y**2=16

from sympy import plot_implicit,symbols,Eq
x,y=symbols('x,y')
p1=plot_implicit(Eq(x**2+y**2,16),(x,-16,16),(y,-16,16);
title=("circle")


6. ( B )     x1+2x2-x3=0,2x1+x2+4x3=0,3x1+3x2+4x3=0

import numpy as np
A=np.matrix([[1,2,-1],[2,1,4],[3,3,4]])
B=np.matrix([[0],[0],[0]])
r=np.linalg.matrix_rank(A)
n=A.shape[1]
if(r==n):
    print("System has trivial solution")
else:
    print('System has',n-r,'non - trivial solution(s)')


7. ( A )       maclaurins series sin(x)+cos(x) upto 3rd degree term

import numpy as np
from sympy import *
x=Symbol('x')
y=sin(x)+cos(x)
a=float(0)
y1=diff(y,x)
y2=diff(y1,x)
y3=diff(y2,x)
f=lambdify(x,y)
f1=lambdiify(x,y1)
f2=lambdiify(x,y2)
f3=lambdiify(x,y3)
y=f(a)+((x-a)/1)*f1(a)+((x-a)**2/2)*f2(a)+((x-a)**3/6)*f3(a)
print(simplify(y))



7. ( B )       integral 0-3,0-3-x,0-3-x-y (xyz) dxdy dz

from sympy import *
x=symbol ('x')
y=symbol ('y')
z=symbol ('z')
w2=integrate((x*+y*+z),(2,0,3-x-y),(y,0,3-x),(x,0,3))
print(w2)





8. ( A )      d(x,y,z) / d(p,o/,0) jacobian

from sympy import *
from sympy.abc import rho,phi,theta
X=rho*cos(phi)*sin(theta);
Y=rho*cos(phi)*cos(theta);
Z=rho*sin(phi);
dx=Derivative(X,rho).doit()
dy=Derivative(Y,rho).doit()
dz=Derivative(Z,rho).doit()
dx1=Derivative(X,phi).doit()
dy1=Derivative(Y,phi).doit()
dz1=Derivative(Z,phi).doit()
dx2=Derivative(X,theta).doit()
dy2=Derivative(Y,theta).doit()
dz2=Derivative(Z,theta).doit()
J=Matrix([[dx,dy,dz],[dx1,dy1,dz1],[dx2,dy2,dz2]])
print('the jacobian matrix is')
display(j)
print("j=")
display(simplify(Determinant(j).doit()))



8. ( B )        iinttegral 0-1,0-x (x**2+y**2)dy dx

from sympy import*
x=symbol('x')
y=symbol('y')
w1=integrate ((x**2+y**2),(y,0,x),(x,0,1))
print(simplify(w1))

































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

