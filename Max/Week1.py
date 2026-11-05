import numpy as np
import matplotlib.pyplot as plt

# Define the potential
def V(r):
    e1 = np.exp(-1/2*(r[0]**2+(r[1]+1.5)**2)/(0.8**2))
    e2 = np.exp(-1/2*((r[0]-0.5)**2+(r[1]-1)**2)/(0.8**2))
    return 2*e1 + 1.5*e2

# Calculate the gradient of the potential (worked out by hand)
def gradV(r):
    e1 = np.exp(-1/2*(r[0]**2+(r[1]+1.5)**2)/(0.8**2))
    e2 = np.exp(-1/2*((r[0]-0.5)**2+(r[1]-1)**2)/(0.8**2))

    return np.array([-2*r[0]/(0.8**2)*e1 -1.5*(r[0]-0.5)/(0.8**2)*e2, 
    -2*(r[1]+1.5)/(0.8**2)*e1 -1.5*(r[1]-1)/(0.8**2)*e2])

# Define the right hand side function f(y) for RK4
def f(y):
    return np.concatenate((y[2:], - gradV(y[:2])))

# Use the RK4 function for r=(x,y), v=(vx,vy), t and dt to find rn and vn for each step n>1 until tn>=5. Also, plots x 
# against y for each rn =(xn,yn)
def RK4(r0, v0, t0, dt):
    # y = [x,y,vx,vy]
    yn = np.concatenate((r0,v0))
    tn = t0
    rns, vns, tns = [], [], []

    while tn < 5:
        rns.append(yn[:2])
        vns.append(yn[2:])
        tns.append(tn) 

        k1 = f(yn)
        k2 = f(yn + k1*dt/2)
        k3 = f(yn + k2*dt/2)
        k4 = f(yn + k3*dt)

        yn = yn + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        tn = tn + dt

    rns = np.array(rns)
    xs = rns[:,0]
    ys = rns[:,1]

    plt.scatter(xs,ys,s=1)
    return rns, vns, tns

L = 3
t0 = 0
dt = 0.01

'''
r0 = np.array([-3, 0.5])
v0 = np.array([2, 0])
RK4(r0, v0, t0, dt)
'''

# Create 'random' r0s and v0s and then plot them by using RK4 function (Added after week 1)
rng = np.random.default_rng(42)
N = 20
r0s = np.array([np.ones(N) * -L, rng.uniform(-L,L,N)]).T
v0s = np.array([rng.uniform(0.5,3,N), rng.uniform(-3,3,N)]).T

for i in range(20):
    RK4(r0s[i], v0s[i], 0, 0.01)

# Contour plot of the potential, V
x = np.linspace(-4,4,400)
y = np.linspace(-4,4,400)
X, Y = np.meshgrid(x,y)
Z = V(np.array([X,Y]))

n = 0.2
ls = []
while n<2:
    ls.append(n)
    n += 0.2

plt.contour(X,Y,Z, levels=ls)

plt.xlim([-3,5])
plt.ylim([-4,4])
plt.show()
