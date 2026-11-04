import numpy as np
import matplotlib.pyplot as plt
r0=[-5,3]
v0=[2,0]
def pot(rn):
    x=rn[0]
    y=rn[1]
    V=2*np.exp(-(0.5)*(x**2+(y+1.5)**2)/0.64)+1.5*np.exp(-(0.5)*((x-0.5)**2+(y-1)**2)/0.64)
    return(V)

def f(state):
    r = state[:2]      
    v = state[2:]      

    eps = 1e-5
    
    dVdx = (pot([r[0] + eps, r[1]])
            - pot([r[0] - eps, r[1]])) / (2 * eps)

    dVdy = (pot([r[0], r[1] + eps])
            - pot([r[0], r[1] - eps])) / (2 * eps)

    a = np.array([-dVdx, -dVdy])

    return np.concatenate((v, a))
def rk4_step(state, dt):
    k1 = f(state)
    k2 = f(state + dt*k1/2)
    k3 = f(state + dt*k2/2)
    k4 = f(state + dt*k3)

    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

dt = 0.01
T = 8
N = round(T / dt)

y_values = np.arange(-4, 5, 1)

plt.figure(figsize=(10, 7))

for y0 in y_values:
    state = np.array([-5, y0, 2, 0], dtype=float)
    trajectory = [state.copy()]
    for i in range(N):
        state = rk4_step(state, dt)
        trajectory.append(state.copy())
    trajectory = np.array(trajectory)
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label=f"y₀ = {y0}"
    )
    plt.scatter(-5, y0, s=20)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle trajectories for different initial y positions")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

#Image generated seem to work- ie deflections make logical sense, higher deflection angle closer to centres
#also playing about with it it seems to work when changing number of particles, potential function etc
# only bit that I think I now understand (and previously didn't) is the rk4, but after some explanation and research it's okay now

