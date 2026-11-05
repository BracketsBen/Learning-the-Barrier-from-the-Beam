import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

# Known function f(x)=sin(x) + cos(x^2)
def f(x):
    return jnp.sin(x) + jnp.cos(x**2)

# Producing N, random data points for each x in [-pi, pi] and y = sin(x) + cos(x^2)
def data_points(N):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-jnp.pi, jnp.pi, N)
    ys = f(xs)
    noisy_ys = ys + rng.normal(0, 0.1, N)
    return(xs,noisy_ys)

# Initialising parameters for Weights and biasses depending on the size of each layer and number of layers
def init_params(sizes, key):
    out = []
    for i in range(len(sizes) - 1):
        key, subkey = jax.random.split(key)
        scale = jnp.sqrt(1/sizes[i])
        W = jax.random.normal(subkey, (sizes[i], sizes[i+1])) * scale
        b = jnp.zeros(sizes[i+1])
        out.append((W,b))
    return out

# Using activator function, tanh and Weights and biasses to return the output of an input
def mlp(params, inp):
    for W,b in params[:-1]:
        inp = jnp.tanh(inp @ W + b)
    W,b = params[-1]
    return inp @ W + b

# Predicting the output of one value, x with parameters
def predict_one(x, params):
    return mlp(params, jnp.array([x]))[0]

# Returning the loss of parameters compared to data points
def loss(params):
    return jnp.mean((predict(xs,params) - ys) ** 2)

# Step function which takes current parameters and optimisation state to return updated parameters,
# optimisation state and loss value from back propagation
@ jax.jit # Compiles the whole function once which speeds the function up
def step(params, opt_state):
    loss_and_grad = jax.value_and_grad(loss)
    loss_val, grads = loss_and_grad(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

seed = 42
key = jax.random.PRNGKey(seed)
#k_data, k_params = jax.random.split(key) (Initially used for data points rather than rng)

xs,ys = data_points(50)

params = init_params([1,16,1], key)

predict = jax.vmap(predict_one, in_axes=(0, None))
predictions = predict(xs, params)

learning_rate = 0.01
opt = optax.adam(learning_rate)
opt_state = opt.init(params)

losses = []

for i in range(3001):
    params, opt_state, loss_val = step(params, opt_state)
    losses.append(loss_val)
    if i % 300 == 0:
       print(i, loss_val)

x_axis = jnp.linspace(-jnp.pi, jnp.pi, 200)
y_axis = predict(x_axis, params)

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
