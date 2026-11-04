import numpy as np
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
import random

# Define the potential - this is the one from the notes
def V(x, y):
    return (2 * np.exp(-0.5*(x**2 + (y + 1.5)**2) / (0.8**2))
            + 1.5 * np.exp(-0.5*((x - 0.5)**2 + (y - 1)**2) / (0.8**2)))

# Calculate the derivatives by hand
def dVdx(x, y):
    return ( (-2 * x /(0.8**2)) * np.exp(-0.5*(x**2 + (y+1.5)**2) / (0.8**2))
            - (1.5 * (x-0.5) /(0.8**2) * np.exp(-0.5*((x-0.5)**2 + (y-1)**2) / (0.8**2))))
def dVdy(x, y):
    return ( (-2 * (y+1.5) /(0.8**2)) * np.exp(-0.5*(x**2 + (y+1.5)**2) / (0.8**2))
            - (1.5 * (y-1) /(0.8**2) * np.exp(-0.5*((x-0.5)**2 + (y-1)**2) / (0.8**2))))

#  Create the 4th order Runge-Kutta method
def RK4(x, y, v_x, v_y, h):
    # k1
    k1_1 = v_x
    k1_2 = v_y
    k1_3 = -dVdx(x, y)
    k1_4 = -dVdy(x, y)

    # k2
    k2_1 = v_x + (0.5 * h * k1_3)
    k2_2 = v_y + (0.5 * h * k1_4)
    k2_3 = -dVdx(x + (0.5 * k1_1 * h), y + (0.5 * h * k1_2))
    k2_4 = -dVdy(x + (0.5 * k1_1 * h), y + (0.5 * h * k1_2))

    # k3
    k3_1 = v_x + (0.5 * h * k2_3)
    k3_2 = v_y + (0.5 * h * k2_4)
    k3_3 = -dVdx(x + (0.5 * k2_1 * h), y + (0.5 * h * k2_2))
    k3_4 = -dVdy(x + (0.5 * k2_1 * h), y + (0.5 * h * k2_2))

    # k4
    k4_1 = v_x + (h * k3_3)
    k4_2 = v_y + (h * k3_4)
    k4_3 = -dVdx(x + (0.5 * k3_1 * h), y + (0.5 * h * k3_2))
    k4_4 = -dVdy(x + (0.5 * k3_1 * h), y + (0.5 * h * k3_2))

    # Update x, y, vx and vy
    x = x + (h / 6) * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
    y = y + (h/ 6) * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
    v_x = v_x + (h / 6) * (k1_3 + 2*k2_3 + 2*k3_3 + k4_3)
    v_y = v_y + (h / 6) * (k1_4 + 2*k2_4 + 2*k3_4 + k4_4)

    return x, y, v_x, v_y
  
# Create contour potential lines to plot
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-4, 4, 200)
contourplot = []
for y in y_grid:
    row = []
    for x in x_grid:
        row = row + [V(x, y)]
    contourplot = contourplot + [row]
       
# Plot figure    
plt.figure
plt.contour(x_grid, y_grid, contourplot, levels=8, colors="black")

for yvalues in np.linspace(-4, 4, 15):
    h = 0.01
    L = 5
    # initial positions
    x = -L
    y = yvalues
    # initial horizontal and vertical velocities 
    v_x = 2.5
    v_y = random.uniform(-1,1)
    
    x_values = []
    y_values = []

    while x < L:
        # add current x and y values
        x_values = x_values + [x]
        y_values = y_values + [y]
        # use RK4 to update next x,y and velcitiy values
        x, y, v_x, v_y = RK4(x, y, v_x, v_y, h)
    # plot the path        
    plt.plot(x_values, y_values, color="red")


plt.xlim(-5,5)
plt.ylim(-4,4)
plt.grid()
plt.show()
