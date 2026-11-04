import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# define g and length
g = 9.81
l = 1.0

# time restriction and step number
t = 5
steps = 500
dt =t/steps

# Initial conditions
theta0 = 1.0     
omega0 = 0.0      

#this is just defining a vector differentiated from the state vector
def rhs(z, l):
    theta, omega = z
    dtheta = omega
    domega = -(g / l) * jnp.sin(theta)
    return jnp.array([dtheta, domega])

#this is just the rk4 code I've been using, it runs cleanly
def rk4_step(z, l):
    k1 = rhs(z, l)
    k2 = rhs(z + 0.5 * dt * k1, l)
    k3 = rhs(z + 0.5 * dt * k2, l)
    k4 = rhs(z + dt * k3, l)

    return z + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )

#just to compile for speed
@jax.jit
#this is the bit I think is broken, I've tried to explain each small section to help me fix it but I don't know how :()
def trajectory(l, theta0, omega0):
    #state vector
    z0 = jnp.array([theta0, omega0])
    #run rk4 on the current state
    def step(z, _):
        z_new = rk4_step(z, l)
        return z_new, z_new
    #used gAI to explain this/understand JAX- apparently it needed the return twice
    #this next part is the code I didn't write and believe to be the problem
    _, states = jax.lax.scan(
        step, z0, None, length=steps
    )
    states = jnp.concatenate([
        z0[jnp.newaxis, :],
        states
    ])
    return states


#the rest is just the simulation
states = trajectory(l, theta0, omega0)

times = jnp.linspace(0.0, t,steps + 1)
thetas = states[:, 0]
omegas = states[:, 1]

final_theta = thetas[-1]
final_omega = omegas[-1]

print(f"Pendulum length: {l:.3f} m")
print(f"Final angle after {t:.1f} s: {float(final_theta):.6f} rad")
print(f"Final angular velocity: {float(final_omega):.6f} rad/s")

# Plot angle against time
plt.figure(figsize=(10, 5))
plt.plot(times, thetas, label=r"$\theta(t)$")
plt.xlabel("Time (s)")
plt.ylabel("Angle (radians)")
plt.title("Pendulum Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#this is all I have, and I'm trying to wrap my head around it but for some reason it's not clicking
#if I simulate it for longer periods of time, it just keep increasing to unrealistic angles and velocities, so something must be wrong
#
