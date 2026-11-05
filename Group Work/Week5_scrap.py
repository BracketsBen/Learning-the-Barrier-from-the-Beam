import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# 1 - Setup 
'''def rhs(state):
    x, vx = state
    return jnp.array([vx, 0.0])

def rk4_step(state):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * DT * k1)
    k3 = rhs(state + 0.5 * DT * k2)
    k4 = rhs(state + DT * k3)
    return state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def trajectory(state0):
    def step(state, _):
        new_state = rk4_step(state)
        return new_state, new_state[0] # (carry, stored output)

    final_state, xs = jax.lax.scan(step, state0, None, length=N_STEPS)
    return jnp.concatenate([state0[:1], xs])'''

# 2 - True potentials and force
def V_true(x,y):
    e1 = jnp.exp(-1/2*(x**2+(y+1.5)**2)/(0.8**2))
    e2 = jnp.exp(-1/2*((x-0.5)**2+(y-1)**2)/(0.8**2))
    return 2*e1 + 1.5*e2

dVdx = jax.grad(V_true, argnums=0)
dVdy = jax.grad(V_true, argnums=1)

def Force(x,y):
    return jnp.array([-dVdx(x,y), -dVdy(x,y)])

# 3 - Integrator
def rhs_fn(state):
    x,y = state[0], state[1]
    v = state[2:]
    a = Force(x, y) # Assuming m = 1
    return jnp.concatenate((v, a))

def rk4_step(state, rhs_fn):
    k1 = rhs_fn(state)
    k2 = rhs_fn(state + 0.5 * DT * k1)
    k3 = rhs_fn(state + 0.5 * DT * k2)
    k4 = rhs_fn(state + DT * k3)
    return state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def trajectory(y0, vy0, rhs_fn):
    state0 = jnp.array([-L, y0, vx0, vy0])
    def step(state, _):
        new_state = rk4_step(state, rhs_fn)
        return new_state, new_state
    final_state, traj = jax.lax.scan(step, state0, None, length = N_STEPS)
    return jnp.concatenate([state0.reshape(-1,4), traj])

# 4 - Exit data
def exit_state_from_traj(traj):
    i = jnp.argmax(traj[:,0]>=L)
    prev, current = traj[i-1], traj[i]
    frac = (L-prev[0])/(current[0]-prev[0])
    y = prev[1] + frac * (current[1]-prev[1])
    vy = prev[3] + frac * (current[3]-prev[3])
    return jnp.array([y, vy])

# 5 - Observations
def true_exit(y0, vy0):
    traj = trajectory(y0, vy0, rhs_fn)
    exit_state = exit_state_from_traj(traj)
    return exit_state

# 6 - Network potential
def init_params(sizes):
    out = []
    for i in range(len(sizes) - 1):
        key, subkey = jax.random.split(k_params)
        scale = jnp.sqrt(1/sizes[i])
        W = jax.random.normal(subkey, (sizes[i], sizes[i+1])) * scale
        b = jnp.zeros(sizes[i+1])
        out.append((W,b))
    return out

def mlp(params, inp):
    for W,b in params[:-1]:
        inp = jnp.tanh(inp @ W + b)
    W,b = params[-1]
    return inp @ W + b

def V_net(p_V, x, y):
    inp = jnp.stack([x / L, y / L]) # Division to keep values in (-1,1) for activaton function
    raw = mlp(p_V, inp)[0] # Keeping V > 0
    return jax.nn.softplus(raw - 2.0) # -2 keeps V low initially

n_dVdx = jax.grad(V_net, argnums=1)
n_dVdy = jax.grad(V_net, argnums=2)

def F_net(x,y):
    return jnp.array([-n_dVdx, -n_dVdy])



L = 5.0 # x0
vx0 = 3.0 # f(x,t)
T = 4.5
N_STEPS = 80
DT = T/N_STEPS

x,y = [0.0,0.0]
state = jnp.array([-L, 3.0, vx0, 1.0])

traj = trajectory(state[1], state[3], rhs_fn)

#plt.plot(traj[:, 0], traj[:, 1])
#plt.show()

# 5 - Observations
seed = 42
key = jax.random.PRNGKey(seed)
k_ys, k_vys, k_noise, k_params = jax.random.split(key, 4)

y0s = jax.random.uniform(k_ys, 100, minval=-3.5, maxval=3.5)
vy0s = jax.random.uniform(k_vys, 100, minval=-1.2, maxval=1.2)

mapping = jax.vmap(true_exit, in_axes=(0,0))
exit_ys, exit_vys = jnp.transpose(mapping(y0s,vy0s))

kn_y, kn_vy = jax.random.split(k_noise)
noise_y = (jnp.max(exit_ys) - jnp.min(exit_ys)) * 0.01
noise_vy = (jnp.max(exit_vys) - jnp.min(exit_vys)) * 0.01
ys = exit_ys + jax.random.normal(kn_y, 100) * noise_y
vys = exit_vys + jax.random.normal(kn_vy, 100) * noise_vy

# 6 - Network Potential
p_V = init_params([2,24,24,1])
