import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

def f(x):
    return jnp.sin(x) + jnp.cos(x**2)

def data_points(N):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-jnp.pi, jnp.pi, N)
    ys = f(xs)
    noisy_ys = ys + rng.normal(0, 0.1, N)
    return(xs,noisy_ys)

def init_params(sizes, key):
    out = []
    for i in range(len(sizes) - 1):
        key, subkey = jax.random.split(key)
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

def predict_one(x, params):
    return mlp(params, jnp.array([x]))[0]

def loss(params):
    return jnp.mean((predict(xs,params) - ys) ** 2)

@ jax.jit
def step(params, opt_state):
    loss_and_grad = jax.value_and_grad(loss)
    loss_val, grads = loss_and_grad(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

seed = 42
key = jax.random.PRNGKey(seed)
k_data, k_params = jax.random.split(key)

xs,ys = data_points(50)

params = init_params([1,16,1], key)

predict = jax.vmap(predict_one, in_axes=(0, None))
predictions = predict(xs, params)

learning_rate = 0.015
opt = optax.adam(learning_rate)
opt_state = opt.init(params)

losses = []
for i in range(2001):
    params, opt_state, loss_val = step(params, opt_state)
    losses.append(loss_val)
    if i % 300 == 0:
        print(i, loss_val)

x_axis = jnp.linspace(-jnp.pi, jnp.pi, 200)
y_axis = predict(x_axis, params)

print(jnp.mean((y_axis-f(x_axis)) ** 2))

plt.plot(x_axis, y_axis, color='blue')
plt.plot(x_axis, f(x_axis), color='green')
plt.scatter(xs, ys, color='red', marker='x')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.loglog(losses)
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()
