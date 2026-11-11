import jax
import jax.numpy as jnp

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

def trajectory(ell, theta0): # Gives trajectory in thetas for a pendulum with length ell and initial position theta0
    z0 = jnp.array([theta0, 0.0])

    def step(z, _): # Returns the carry and stored output as the function for each step for jax.lax.scan
        z_new = rk4_step(z, ell)
        return z_new, z_new[0] # (carry, stored output)   

    _, thetas = jax.lax.scan(step, z0, None, length=N_STEPS)
    return jnp.concatenate([jnp.array([theta0]), thetas])

def final_angle(ell, theta0): # Returns the final angle for the trajectory of pendulum with length ell and initial position theta0
    return trajectory(ell, theta0)[-1]

def final_angle_lam(lam, theta0):
    ell = g / lam
    return final_angle(ell, theta0)

def final_angle_derivative(ell, theta0): # Returns the derivative with respect to lambda=g/ell at final time T
    lam = g / ell
    return (final_angle_lam(lam + h, theta0) - final_angle_lam(lam, theta0)) / h

def derivative_lam(lam, theta0):
    return (final_angle_lam(lam + h, theta0) - final_angle_lam(lam, theta0)) / h

def find_lam(theta0, thetaf, max_iterations): # Uses Newton's root finding algorithm to find a close value for ell such that the 
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

lam, theta0 = g / 3.0, jnp.pi / 2 # Checking the difference between jax.grad and finite derivatives
print(dfinal_dlam(lam, theta0))
print(derivative_lam(lam, theta0))
