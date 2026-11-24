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

# 3 - Integrator- Asa Chapman
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

# 6 - Network potential
# Creates initial weights and biases
def init_params(sizes):
    out = []
    key = jax.random.PRNGKey(32)
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(1.0 / sizes[i])
        W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
        b = jnp.zeros(sizes[i + 1])
        out.append((W, b))
    return out

LAYERS = [2, 24, 24, 1]
p_V = init_params(LAYERS)

# Calculates the output based on weights and biases
def mlp(parameters, inp):
    for i in range(len(parameters)):
        W, b = parameters[i]
        inp = jnp.dot(inp, W) + b
        if i < len(parameters)-1:
            inp = jnp.tanh(inp)      
    return inp

# Calculates the potential at (x,y) based on the weights and biases
def V_net(p_V, x, y):
    inp = jnp.stack([x / L, y / L])
    raw = mlp(p_V, inp)[0]
    return jax.nn.softplus(raw - 2.0)

dVnetdx = jax.grad(V_net, argnums = 1)
dVnetdy = jax.grad(V_net, argnums = 2)
def force_net(p_V,x,y):
    return jnp.array([-dVnetdx(p_V,x,y), -dVnetdy(p_V,x,y)])

# 7 - Predictions
# Predict the exit position and vertical velocity for one particle
def predict_exit(y0, vy0, p_V):
    
    def rhs_fn_net(state):
        x, y, vx, vy = state
        return jnp.array([vx, vy, -dVnetdx(p_V,x,y), -dVnetdy(p_V,x,y)])
    
    trajectory = integrate_traj(y0, vy0, rhs_fn_net)
    return exit_state_from_traj(trajectory)

# Predict the exit data for all particles
predict_exitvalues_allparticles = jax.vmap(predict_exit, in_axes=(0, 0, None))
predicted_exitvalues_allparticles = predict_exitvalues_allparticles(y0_values, vy0_values, p_V)

# 8 - Loss
def loss_fn(p_V):
    predicted_exitvalues_allparticles = predict_exitvalues_allparticles(y0_values, vy0_values, p_V)
    predicted_y = predicted_exitvalues_allparticles[:, 0]
    predicted_vy = predicted_exitvalues_allparticles[:, 1]
    
    mse_y = jnp.mean((predicted_y - observed_y_exit_values)**2)
    mse_vy = jnp.mean((predicted_vy - observed_vy_exit_values)**2)

    return 0.5 * ( (mse_y / (noise_y**2)) + (mse_vy / (noise_vy**2)) )

# Calculate the loss and its gradients
loss_value_and_grad = jax.value_and_grad(loss_fn)
loss_value, gradients = loss_value_and_grad(p_V)

#9. Optimiser

schedule = optax.exponential_decay(
    init_value=1e-2,
    transition_steps=600,
    decay_rate=1e-2
)

clip = optax.clip_by_global_norm(0.1)

opt = optax.chain(clip, optax.adam(schedule))
  
opt_state = opt.init(p_V)

# 10. Training- Asa Chapman

#optimiser state
opt_state = opt.init(p_V)

#def train step
@jax.jit
def train_step(p_V, opt_state):

    #loss
    loss, gradients = jax.value_and_grad(loss_fn)(p_V)

    #update params
    updates, opt_state = opt.update(gradients, opt_state, p_V)
    p_V = optax.apply_updates(p_V, updates)

    return p_V, opt_state, loss


#600 iterations
loss_history = []

for i in range(600):
    p_V, opt_state, loss = train_step(p_V, opt_state)
    loss_history.append(float(loss))

#plot loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.show()


# 11. Diagnostics- Asa Chapman

#make grid
x_values = jnp.linspace(-L, L, 100)
y_values = jnp.linspace(-L, L, 100)

X, Y = jnp.meshgrid(x_values, y_values)

#evaluate potential
V_true_grid = jax.vmap(jax.vmap(V_true))(X, Y)

#evaluate network potential
V_net_grid = jax.vmap(
    jax.vmap(lambda x, y: V_net(p_V, x, y))
)(X, Y)

#calc difference
V_difference = V_net_grid - V_true_grid

#numpy conversion
X_plot = np.array(X)
Y_plot = np.array(Y)
V_true_grid = np.array(V_true_grid)
V_net_grid = np.array(V_net_grid)
V_difference = np.array(V_difference)


#plot both potentials
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

plots = [
    (V_true_grid, "True Potential"),
    (V_net_grid, "Reconstructed Potential"),
    (V_difference, "Difference")
]

for ax, (data, title) in zip(axes, plots):
    im = ax.pcolormesh(X_plot, Y_plot, data, shading="auto")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


#compare predicted and observed data
predicted_exit_values = predict_exitvalues_allparticles(
    y0_values, vy0_values, p_V
)

predicted_y = np.array(predicted_exit_values[:, 0])
predicted_vy = np.array(predicted_exit_values[:, 1])

observed_y = np.array(observed_y_exit_values)
observed_vy = np.array(observed_vy_exit_values)

#plot exit pos and v
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(observed_y, predicted_y)

low = min(observed_y.min(), predicted_y.min())
high = max(observed_y.max(), predicted_y.max())

axes[0].plot([low, high], [low, high], "r--")
axes[0].set_xlabel("Observed Exit Position")
axes[0].set_ylabel("Predicted Exit Position")
axes[0].set_title("Exit Position Comparison")

axes[1].scatter(observed_vy, predicted_vy)

low = min(observed_vy.min(), predicted_vy.min())
high = max(observed_vy.max(), predicted_vy.max())

axes[1].plot([low, high], [low, high], "r--")
axes[1].set_xlabel("Observed Exit Velocity")
axes[1].set_ylabel("Predicted Exit Velocity")
axes[1].set_title("Exit Velocity Comparison")

plt.tight_layout()
plt.show()


#define motion of trained network
def rhs_fn_net(state):
    x, y, vx, vy = state

    return jnp.array([
        vx,
        vy,
        -dVnetdx(p_V, x, y),
        -dVnetdy(p_V, x, y)
    ])


#plot 10 particles
plt.figure(figsize=(10, 7))

for i in range(10):
    true_trajectory = integrate_traj(
        y0_values[i], vy0_values[i], rhs_fn
    )
    predicted_trajectory = integrate_traj(
        y0_values[i], vy0_values[i], rhs_fn_net
    )

    true_trajectory = np.array(true_trajectory)
    predicted_trajectory = np.array(predicted_trajectory)

    line, = plt.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        label=f"Particle {i+1} True"
    )

    plt.plot(
        predicted_trajectory[:, 0],
        predicted_trajectory[:, 1],
        "--",
        color=line.get_color(),
        label=f"Particle {i+1} Predicted"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("True vs Predicted Particle Trajectories")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid()
plt.tight_layout()
plt.show()
