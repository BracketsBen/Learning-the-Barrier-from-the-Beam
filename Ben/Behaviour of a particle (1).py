import numpy as np
import matplotlib.pyplot as plt
import random

# Potential
def V(x, y):
    return (
        2 * np.exp(-0.5*(x**2 + (y + 1.5)**2) / (0.8**2))
        + 1.5 * np.exp(-0.5*((x - 0.5)**2 + (y - 1)**2) / (0.8**2))
    )


# Derivative with respect to x
def dVdx(x, y):
    return (
        (-2 * x / (0.8**2)) *
        np.exp(-0.5*(x**2 + (y+1.5)**2) / (0.8**2))
        - (1.5 * (x-0.5) / (0.8**2) *
        np.exp(-0.5*((x-0.5)**2 + (y-1)**2) / (0.8**2)))
    )


# Derivative with respect to y
def dVdy(x, y):
    return (
        (-2 * (y+1.5) / (0.8**2)) *
        np.exp(-0.5*(x**2 + (y+1.5)**2) / (0.8**2))
        - (1.5 * (y-1) / (0.8**2) *
        np.exp(-0.5*((x-0.5)**2 + (y-1)**2) / (0.8**2)))
    )


# Equations of motion
def derivatives(values):
    x, y, v_x, v_y = values

    F_x = -dVdx(x, y)
    F_y = -dVdy(x, y)

    return np.array([v_x, v_y, F_x, F_y])


# RK4
def RK4(values, h):

    k1 = derivatives(values)
    k2 = derivatives(values + (h/2)*k1)
    k3 = derivatives(values + (h/2)*k2)
    k4 = derivatives(values + h*k3)

    return values + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


# Initial conditions
L = 3.0
h = 0.01
T = 10.0

values = np.array([-L, 0.0, 1.0, 0.0])

# Calculate trajectory
trajectory = []
t = 0.0

while t < T and values[0] < L:

    trajectory.append(values.copy())

    values = RK4(values, h)

    t += h

trajectory = np.array(trajectory)

# Create contour potential lines to plot
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-4, 4, 200)
contourplot = []
for y in y_grid:
    row = []
    for x in x_grid:
        row = row + [V(x, y)]
    contourplot = contourplot + [row]
       
# Plot the countour lines 
plt.figure
plt.contour(x_grid, y_grid, contourplot, levels=8, colors="black")



for yvalues in np.linspace(-4, 4, 15):

    L = 5

    x = -L
    y = yvalues

    v_x = 2.5
    v_y = random.uniform(-1, 1)

    values = np.array([x, y, v_x, v_y])

    x_values = []
    y_values = []

    t = 0.0

    while t < T and values[0] < L:

        x_values.append(values[0])
        y_values.append(values[1])

        values = RK4(values, h)

        t += h

    plt.plot(x_values[0], y_values[0], 'o',
         markerfacecolor='none', markeredgecolor='black')

    plt.plot(x_values, y_values, color="red")

plt.plot(x, y, label="Particle trajectory,", color = "red")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle trajectory using RK4", loc = "right")
plt.legend()
plt.grid()
plt.axis("equal")
plt.xlim(-5.2, 5.2)
plt.ylim(-4.2, 4.2)
plt.margins(0)
plt.show()
