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

# Make some example measured points
ell_measured = 5.74
theta0 = 0.2 * jnp.pi
times = jnp.linspace(0, T, 11)
ratio_measured = g / ell_measured

trajectory_measured = trajectory(ell_measured, theta0)
angles_measured = trajectory_measured[jnp.round(times / DT).astype(int)]

# Add some random noise to the angles 
angles_measured = angles_measured + 0.001 * jax.random.normal(jax.random.PRNGKey(0), shape = (11,))

#print("Measured times:", times)
#print("Measured angles:", angles_measured)

# Calculate the error between data and simulated trajectory
def error(ratio):
    ell_guess = g / ratio
    trajectory_simulated = trajectory(ell_guess, theta0)
    angles_simulated = trajectory_simulated[jnp.round(times / DT).astype(int)]
    return jnp.sum((angles_simulated - angles_measured)**2)

# Use the newton method to minimise the error
def newton_method2(ratio_guess, x):
    derivative = jax.grad(error,argnums=0)
    second_derivative = jax.grad(derivative, argnums=0)
    ratio_updateguess = ratio_guess - (derivative(ratio_guess) / second_derivative(ratio_guess))
    return ratio_updateguess, ratio_updateguess

# Initial guesses
ell_guess = 4
ratio_initial = g / ell_guess

ratio_final_guess, ratio_values = jax.lax.scan(newton_method2, ratio_initial, jnp.linspace(0, 10, 50))

# Print results
print('Estimated ell:', g/ratio_final_guess) 
print("Estimated g / ell:", ratio_final_guess)
print("Measured g / ell:", ratio_measured)
print("Error:", error(ratio_final_guess))
