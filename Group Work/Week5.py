import jax.numpy as jnp
import jax

# 1 - Setup
# Defining variables
L = 5.0
vx0 = 3.0
T, N_STEPS = 4.5, 80
DT = T / N_STEPS
V0, F0 = 0.0, jnp.array([0.0, 0.0]) 

def rhs(state): # Defining the rhs function for RK4 (state = [x, vx])
    return jnp.array([state[1], 0]) # V = 0 ==> F = 0 ==> a = 0 by F = -dV/dx

def rk4_step(state): # Define each step of the RK4 function
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * DT * k1)
    k3 = rhs(state + 0.5 * DT * k2)
    k4 = rhs(state + DT * k3)
    return state + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def trajectory(state0): # Finds the trajectory from a starting state, state0
    def step(state, _):
        new_state = rk4_step(state)
        return new_state, new_state[0] # Carry, stored output

    _, traj = jax.lax.scan(step, state0, None, length=N_STEPS)
    return jnp.concatenate([state0[:1], traj]) # xs

# Check:
'''def crossing_time(state0): # Finding stopping time T for x = L through linear interpolation
    traj = trajectory(state0) # It takes T = 3.333... for it to reach x = +L if V = 0 and L = 5 (since V = 0 the velocity will be
                              # constant and therefore crosses at T = 2*L/vx0 (= 10/3 = 3.333...)
    i = jnp.argmax(traj >= L) # T should be larger because RK4 needs another step to keep working
    frac = (L - traj[i-1]) / (traj[i] - traj[i-1])
    T = DT * ((i - 1) + frac)
    return T

state0 = jnp.array([-L,vx0])
print(crossing_time(state0))'''

# 2 - True Potential and Forces
def V_true(x,y): # Defining the true potential from equation 5
    e1 = jnp.exp(-0.5 * (x ** 2 + (y + 1.5) ** 2) / (0.8 ** 2))
    e2 = jnp.exp(-0.5 * ((x - 0.5) ** 2+(y - 1) ** 2) / (0.8 ** 2))
    return 2*e1 + 1.5*e2

# Finding gradients and therefore, forces
dVdx = jax.grad(V_true, argnums=0)
dVdy = jax.grad(V_true, argnums=1)

def force(x,y):
    return jnp.array([-dVdx(x,y), -dVdy(x,y)])

def dVdx_hand(x,y):
    e1 = jnp.exp(-0.5 * (x ** 2 + (y + 1.5) ** 2) / (0.8 ** 2))
    e2 = jnp.exp(-0.5 * ((x - 0.5) ** 2+(y - 1) ** 2) / (0.8 ** 2))
    return -(2 * x * e1 + 1.5 * (x - 0.5) * e2) /(0.8 ** 2)
def dVdy_hand(x,y):
    e1 = jnp.exp(-0.5 * (x ** 2 + (y + 1.5) ** 2) / (0.8 ** 2))
    e2 = jnp.exp(-0.5 * ((x - 0.5) ** 2+(y - 1) ** 2) / (0.8 ** 2))
    return -(2 * (y + 1.5) * e1 + 1.5 * (y - 1) * e2) / (0.8 ** 2)

x, y = [1.0,3.0]
print("Jax ∂V/∂x: ", dVdx(x,y),"Hand ∂V/∂x", dVdx_hand(x,y), "Jax ∂V/∂y: ", dVdy(x,y), "Hand ∂V/∂y: ", dVdy_hand(x,y))
print(force(x,y))
