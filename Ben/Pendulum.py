import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Constants
g = 9.81
T = 5.0
N_STEPS = 500
DT = T / N_STEPS

h = 0.01

ell = 1

# Equations of motion
def rhs(z, alpha):
    theta, omega = z
    return jnp.array([
        omega,
        -alpha * jnp.sin(theta)
    ])


# One Runge-Kutta 4 step
def rk4_step(z, alpha):
    k1 = rhs(z, alpha)
    k2 = rhs(z + 0.5 * DT * k1, alpha)
    k3 = rhs(z + 0.5 * DT * k2, alpha)
    k4 = rhs(z + DT * k3, alpha)

    return z + (DT / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )


# Calculate the trajectory
def trajectory(alpha, theta0):
    z0 = jnp.array([theta0, 0.0])

    def step(z, _):
        z_new = rk4_step(z, alpha)
        return z_new, z_new[0]

    _, thetas = jax.lax.scan(
        step, z0, None, length=N_STEPS
    )

    return jnp.concatenate([
        jnp.array([theta0]), thetas
    ])


# Return the final angle at time T
def final_angle(alpha, theta0):
    return trajectory(alpha, theta0)[-1]


# Finite difference derivative
def deriv(ratio, theta0):
    theta_t = final_angle(ratio, theta0)

    theta_th = final_angle(ratio + h, theta0)

    derivative = (theta_th - theta_t) / h

    return derivative


# Analytic  derivative using JAX
def autodiff_deriv(ratio, theta0):
    return jax.grad(final_angle, argnums=0)(ratio, theta0)


# Initial conditions
ratio = g / ell
theta0 = 0.2

# Calculate results
print("Final angle:", final_angle(ratio, theta0))

print("Finite difference derivative:",
      deriv(ratio, theta0))

print("JAX derivative:",
      autodiff_deriv(ratio, theta0))

