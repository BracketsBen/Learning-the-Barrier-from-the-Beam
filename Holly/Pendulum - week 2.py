import jax
import jax.numpy as jnp

g = 9.81
T = 5.0
N_STEPS = 500
DT = T / N_STEPS

# Equations of motion
def rhs(z, ell):
    theta, omega = z
    return jnp.array([omega, -(g / ell) * jnp.sin(theta)])

def rk4_step(z, ell):
    k1 = rhs(z, ell)
    k2 = rhs(z + 0.5 * DT * k1, ell)
    k3 = rhs(z + 0.5 * DT * k2, ell)
    k4 = rhs(z + DT * k3, ell)
    return z + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Use the RK4 to calculate the motion of the pendulum
def trajectory(ell, theta0):
    z0 = jnp.array([theta0, 0.0])

    def step(z, _):
        z_new = rk4_step(z, ell)
        return z_new, z_new[0] # (carry, stored output)
    
    _, thetas = jax.lax.scan(step, z0, None, length=N_STEPS)
    return jnp.concatenate([jnp.array([theta0]), thetas])

# Gives the angle after T seconds
def final_angle(ell, theta0):
    return trajectory(ell, theta0)[-1]

# Choose values
ell = 4
theta0 = 0.3 * jnp.pi
# Calculate the final angle
theta_T = final_angle(ell, theta0)
print('After T seconds, the angle is', theta_T/ jnp.pi,'π')

# Calculate the finite-difference derivative
h = 1e-3
derivative = (final_angle(ell - ((h*ell**2)/(g+h*ell)), theta0) - theta_T) / h
ratio = g / ell

def final_ratio(ratio, theta0):
    ell = g / ratio
    return final_angle(ell, theta0)

# Use the grad operator to calculate the derivative
grad_derivative = jax.grad(final_ratio, argnums=0)

# Compare the finite-difference and grad derivatives
print('After T seconds, the finite-difference derivative is', derivative)
print('After T seconds, the grad derivative is', grad_derivative(ratio, theta0))
print('The difference between the grad derivative and the finite-difference derivative is',abs(grad_derivative(ratio, theta0) - derivative))

theta_f = 0.1 * jnp.pi
ell_initial = ell

# Use the newton method to estimate l if theta = theta_f at time T
def newton_method(ell, x):
    theta_T = final_angle(ell, theta0)
    # Calculate the derivative with respect to g/ell
    derivative = (final_angle(ell - ((h*ell**2)/(g+h*ell)), theta0) - theta_T) / h
    # Update the estimate using Newton's method
    ratio = g / ell
    ratio_new = ratio - (theta_T - theta_f) / derivative
    ell_new = g / ratio_new
    return ell_new, ell_new
  
ell_final, ell_values = jax.lax.scan(newton_method, ell_initial, jnp.linspace(0, 10, 10))
print("To reach θf in T seconds, ell is", ell_final)
