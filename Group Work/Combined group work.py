import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import optax
import numpy as np

# 1 - Setup
L = 5.0
vx0 = 3.0
T, N_STEPS = 4.5, 80
DT = T / N_STEPS

# 2 - True Potential and Forces
# Defining the true potential from equation 5
def V_true(x,y): 
    e1 = jnp.exp(-0.5 * (x ** 2 + (y + 1.5) ** 2) / (0.8 ** 2))
    e2 = jnp.exp(-0.5 * ((x - 0.5) ** 2+(y - 1) ** 2) / (0.8 ** 2))
    return 2*e1 + 1.5*e2

# Finding gradients and forces
dVdx = jax.grad(V_true, argnums=0)
dVdy = jax.grad(V_true, argnums=1)
def force(x,y):
    return jnp.array([-dVdx(x,y), -dVdy(x,y)])

# 3 - Integrator
def rhs_fn(state):
    x, y, vx, vy = state
    return jnp.array([vx, vy, -dVdx(x,y), -dVdy(x,y)])

def rk4_step(state, rhs_fn):
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * DT * k1)
    k3 = rhs_fn(state + 0.5 * DT * k2)
    k4 = rhs_fn(state + DT * k3)

    new_state = state + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
  
# Integrate particle traj over the steps
def integrate_traj(y0, vy0, rhs_fn):
    initial_state = jnp.array([-L, y0, vx0, vy0])

    # Use jax.lax.scan to return all states 
    def step(state, _):
        new_state = rk4_step(state, rhs_fn)
        return new_state, new_state

    _, states = jax.lax.scan(step, initial_state, None, N_STEPS)

    # Add the first state to the trajectory
    trajectory = jnp.concatenate((initial_state[None, :], states), axis=0)
    return trajectory
  
# 4 - Exit data
def exit_state_from_traj(trajectory):
      # Find the index of the first time step when x > L 
      index = jnp.argmax(trajectory[:, 0] >= L)
      x_prev = trajectory[index - 1, 0]
      x_next = trajectory[index, 0]

      # Calculate interpolation ratio
      ratio =  (L - x_prev)/(x_next - x_prev)

      # Interpolate vertical position and velocity
      exit_y = (trajectory[index - 1, 1] * (1-ratio) + ratio * (trajectory[index, 1]))
      exit_vy = (trajectory[index - 1, 3] * (1-ratio) + ratio * (trajectory[index, 3]))
      return jnp.array([exit_y, exit_vy])

# 5 - Observations
# Pick 100 particles 
rng = np.random.default_rng()
y0_values = jnp.asarray(rng.uniform(low = -3.5, high = 3.5, size = 100))
vy0_values = jnp.asarray(rng.uniform(low = -1.2, high = 1.2, size = 100))

def true_exit(y0, vy0):
    trajectory = integrate_traj(y0, vy0, rhs_fn)
    return exit_state_from_traj(trajectory)

# Compute the exit data for the 100 particles
true_exit_values = jax.vmap(true_exit)(y0_values, vy0_values)
true_y_exit_values = true_exit_values[:, 0]
true_vy_exit_values = true_exit_values[:, 1]

# Add noise to the exit data
noise_y = 0.01 * (jnp.max(true_y_exit_values) - jnp.min(true_y_exit_values))
noise_vy = 0.01 * (jnp.max(true_vy_exit_values) - jnp.min(true_vy_exit_values))
noise_yvalues = jnp.asarray(rng.normal(loc = 0.0, scale = noise_y, size = 100))
noise_vyvalues = jnp.asarray(rng.normal(loc = 0.0, scale = noise_vy, size = 100))
observed_y_exit_values = true_y_exit_values + noise_yvalues
observed_vy_exit_values = true_vy_exit_values + noise_vyvalues
