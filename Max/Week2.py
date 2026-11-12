import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Exercise a
g = 9.81
T = 5.0
N_STEPS = 500
DT = T / N_STEPS
h = 1e-3

def rhs(z, ell): # Define the rhs function for RK4
    theta, omega = z
    return jnp.array([omega, -(g / ell) * jnp.sin(theta)])

def rk4_step(z, ell): # Define one step of the RK4
    k1 = rhs(z, ell)
    k2 = rhs(z + 0.5 * DT * k1, ell)
    k3 = rhs(z + 0.5 * DT * k2, ell)
    k4 = rhs(z + DT * k3, ell)
    return z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def trajectory(ell, theta0, N=N_STEPS): # Gives trajectory in thetas for a pendulum with length ell and initial position theta0
    z0 = jnp.array([theta0, 0.0])
    DT = T / N

    def step(z, _): # Returns the carry and stored output as the function for each step for jax.lax.scan
        z_new = rk4_step(z, ell)
        return z_new, z_new[0] # (carry, stored output)   

    _, thetas = jax.lax.scan(step, z0, None, length=N)
    return jnp.concatenate([jnp.array([theta0]), thetas])

def final_angle(ell, theta0): # Returns the final angle for the trajectory of pendulum with length ell and initial position theta0
    return trajectory(ell, theta0, N=N_STEPS)[-1]

def final_angle_lam(lam, theta0):
    ell = g / lam
    return final_angle(ell, theta0)

def final_angle_derivative(ell, theta0): # Returns the derivative with respect to lambda=g/ell at final time T
    lam = g / ell
    return (final_angle_lam(lam + h, theta0) - final_angle_lam(lam, theta0)) / h

def derivative_lam(lam, theta0):
    return (final_angle_lam(lam + h, theta0) - final_angle_lam(lam, theta0)) / h

def find_lam(theta0, thetaf, ell0, max_iterations): # Uses Newton's root finding algorithm to find a close value for ell such that the 
    ell = 3.0                                 # final angle is thetaf
    lam = g / ell
    error = 1e-3
    f = final_angle_lam(lam, theta0) - thetaf
    i = 0
    while f > error and i < max_iterations:
        df = derivative_lam(lam, theta0)
        lam = lam - f / df
        if lam <= 0:
            return 'lam became <= 0'
        f = final_angle_lam(lam, theta0) - thetaf
        i = i + 1
    return lam

# Exercise b
dfinal_dlam = jax.grad(final_angle_lam)

'''lam, theta0 = g / 3.0, jnp.pi / 2 # Checking the difference between jax.grad and finite derivatives
print(dfinal_dlam(lam, theta0))
print(derivative_lam(lam, theta0))'''

# Exercise c
# Creating data points for t in [0.5], ell = 6.35 and theta0 = 0.4 * pi
theta0 = 0.21 * jnp.pi
data = trajectory(0.621, theta0)

'''  # Checks
theta0 = 0.4 * jnp.pi
data = trajectory(501, theta0)
theta0 = 0.21 * jnp.pi
data = trajectory(0.621, theta0)
theta0 = 0.5 * jnp.pi
data = trajectory(15, theta0)
theta0 = 0.4 * jnp.pi
data = trajectory(6.35, theta0)
'''

data_points = data[::50]
ts = jnp.linspace(0, 5, 11)

# Plotting data points from above with theta against t for t in [0,5]
plt.plot(ts, data_points, "x")
plt.xlabel("t")
plt.ylabel(r"$\theta$")

@ jax.jit
def loss(lam): # Finding loss for each lam = g / ell
    lam_data = trajectory(g / lam, theta0, N_STEPS)[::50]
    return jnp.sum((lam_data-data_points) ** 2)

# Finding the minimum loss for some starting lam0 = g / ell0 in [0.5, 5]
lams = jnp.linspace(0.005, 25, 500)
losses = jax.vmap(loss)(lams)
lam0 = lams[jnp.argmin(losses)]

# Defining gradients to be able to Newton root find a good initial lam = g / ell
dloss = jax.grad(loss)
d2losss = jax.grad(dloss)

def init_lam(lam, n=10): # Newton root finding for n iterations for a good intial lam 
    for i in range(n):
        lam = lam - dloss(lam) / d2losss(lam)
    return lam

# Finding lam by initialising it with init_lam for 10 iterations
lam_found = init_lam(lam0, 10)
ell_found = g / lam_found
traj_found = trajectory(ell_found, theta0)

# Plotting theta against t for the trajectory from the lam_found
plt.plot(jnp.linspace(0, 5, 501), traj_found)
plt.show()
print(ell_found)

# Plotting loss for lam = g/ell against each lam
plt.plot(lams, losses)
plt.xlabel(r"$\lambda = \frac{g}{\ell}$")
plt.ylabel("Loss")
plt.show()
