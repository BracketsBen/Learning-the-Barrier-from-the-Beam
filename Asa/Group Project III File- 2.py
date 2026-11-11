
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

#initial values, using alpha as our ratio
g = 9.81
l = 1.0
alpha_true = g / l

#model parameters
t = 5.0
steps = 500
dt = t / steps

#first angle and angular velocity
theta0 = 1.0
omega0 = 0.0

#use state vector and return derivative vector
def rhs(z, alpha):
    theta, omega = z
    dtheta = omega
    domega = -alpha * jnp.sin(theta)
    return jnp.array([dtheta, domega])


#RK4 method, however changed to input our ratio alpha
def rk4_step(z, alpha):
    k1 = rhs(z, alpha)
    k2 = rhs(z + 0.5 * dt * k1, alpha)
    k3 = rhs(z + 0.5 * dt * k2, alpha)
    k4 = rhs(z + dt * k3, alpha)

    return z + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


#same as before, using jax to calculate the steps
@jax.jit
def trajectory(alpha, theta0, omega0):

    z0 = jnp.array([theta0, omega0])

    def step(z, _):
        z_new = rk4_step(z, alpha)
        return z_new, z_new

    _, states = jax.lax.scan(
        step, z0, None, length=steps
    )

    states = jnp.concatenate([
        z0[jnp.newaxis, :],
        states
    ])

    return states


#the actual observed values
true_states = trajectory(alpha_true, theta0, omega0)

true_times = jnp.linspace(0.0, t, steps + 1)

#limiting number of observations 
obs_times = jnp.array([
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
])

#interpolate
obs_thetas = jnp.interp(
    obs_times,
    true_times,
    true_states[:, 0]
)


#again use jax for loss function wrt alpha
@jax.jit
def loss(alpha):

    states = trajectory(alpha, theta0, omega0)

    times = jnp.linspace(0.0, t, steps + 1)

    simulated_thetas = jnp.interp(
        obs_times,
        times,
        states[:, 0]
    )

    error = simulated_thetas - obs_thetas

    return jnp.mean(error**2)


loss_and_grad = jax.jit(jax.value_and_grad(loss))


def objective(x):
    value, grad = loss_and_grad(jnp.asarray(x[0]))

    return float(value), np.array([float(grad)])


#starting g/l (will change with the fct)
initial_guess = np.array([10])

result = minimize(
    objective,
    initial_guess,
    jac=True,
    method="L-BFGS-B",
    bounds=[(0.01, 50.0)]
)

#run the model
estimated_alpha = result.x[0]
estimated_l = g / estimated_alpha

print(f"True g/l: {alpha_true:.6f}")
print(f"Estimated g/l: {estimated_alpha:.6f}")
print(f"True pendulum length: {l:.6f} m")
print(f"Estimated pendulum length: {estimated_l:.6f} m")
print(f"Final fitting error: {result.fun:.10f}")


#use estimated alpha for state vectors
estimated_states = trajectory(
    estimated_alpha, theta0, omega0
)

estimated_times = jnp.linspace(0.0, t, steps + 1)

estimated_thetas = estimated_states[:, 0]


#plot
plt.figure(figsize=(10, 5))

plt.plot(
    estimated_times,
    estimated_thetas,
    label="Estimated trajectory"
)

plt.scatter(
    obs_times,
    obs_thetas,
    color="red",
    label="Observed angles",
    zorder=3
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.title("Pendulum Parameter Estimation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#this is the updated work for 2. playing around with it a tad, it seems to be working just fine. 
#using the state vectors and ratio simplified it a lot (my original code was long and clunky)
#i have uploaded the plots for 2 different initial guesses- any strange guess leads to a mess, so something is going right?
#calculating some values and interpolating also saves time, but I do wonder if this is innacurate- something for Thursday!
