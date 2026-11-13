import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#parameters (although the letters may be different)
L = 5.0
v0 = 3.0
T = 4.5
N_STEPS = 80
DT = T / N_STEPS

def V_true(rn):
    x = rn[0]
    y = rn[1]
    V = 2*jnp.exp(-(0.5)*(x**2+(y+1.5)**2)/0.64) + 1.5*jnp.exp(-(0.5)*((x-0.5)**2+(y-1)**2)/0.64)
    return V


#use jax for gradient of the potential
grad_V_true = jax.grad(V_true)



#time derivative of the state vector
def rhs_fn(state):
    x, y, vx, vy = state

    dVdx, dVdy = grad_V_true(jnp.array([x, y]))

    return jnp.array([
        vx,
        vy,
        -dVdx,
        -dVdy
    ])


#zero potential (no forces acting on the particle)
def rhs_zero(state):
    x, y, vx, vy = state

    return jnp.array([
        vx,
        vy,
        0.0,
        0.0
    ])


#rk4 integration
def rk4_step(state, rhs_fn):

    k1 = rhs_fn(state)

    k2 = rhs_fn(state + 0.5 * DT * k1)

    k3 = rhs_fn(state + 0.5 * DT * k2)

    k4 = rhs_fn(state + DT * k3)

    new_state = state + (DT / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )

    return new_state


#integrate particle traj over the steps
def integrate_traj(y0, vy0, rhs_fn):

    initial_state = jnp.array([-L, y0, v0, vy0])

    #lax.scan to return all states 
    def step(state, _):
        new_state = rk4_step(state, rhs_fn)
        return new_state, new_state

    final_state, states = jax.lax.scan(
        step,
        initial_state,
        None,
        length=N_STEPS
    )

    #just add the first state back
    trajectory = jnp.concatenate(
        (initial_state[None, :], states),
        axis=0
    )

    return trajectory

# code for 4
def exit_state_from_traj(trajectory):
      # Finds the index of crossing 
      index = jnp.argmax(trajectory[:, 0] >= L)
      x_prev = trajectory[index - 1, 0]
      x_next = trajectory[index, 0]

      #Ratio between the steps prior and after

      ratio =  (L - x_prev)/(x_next-x_prev)

      #Interpolates via weighting based on the ratio

      exit_y = (  trajectory[index - 1, 1] *(1-ratio)
        + ratio * (trajectory[index, 1]))
      
      exit_vy = (trajectory[index - 1, 3] *(1-ratio)
        + ratio * (trajectory[index, 3]))

      return jnp.array([exit_y, exit_vy])
    
      
    





#zero potential check
trajectory_zero = integrate_traj(0.0, 1, rhs_zero)
trajectory_zero_part4 = exit_state_from_traj(trajectory_zero)
print(trajectory_zero_part4)

#get position coords
x_zero = trajectory_zero[:, 0]
y_zero = trajectory_zero[:, 1]


#plot it
plt.figure(figsize=(7, 5))
plt.plot(x_zero, y_zero, label="V = 0")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory with Zero Potential")
plt.legend()
plt.grid()
plt.show()

#true potential check
trajectory_true = integrate_traj(0.0, 0.0, rhs_fn)

#get position coords and velocity for true potential
x = trajectory_true[:, 0]
y = trajectory_true[:, 1]
vx = trajectory_true[:, 2]
vy = trajectory_true[:, 3]


#plot it
plt.figure(figsize=(7, 5))
plt.plot(x, y, label="True potential trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle Trajectory Through True Potential")
plt.legend()
plt.grid()
plt.show()
