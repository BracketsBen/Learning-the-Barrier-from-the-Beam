import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax 

import numpy as np



# 1. data

#number of data points of the function
N = 50 

#Standardises the same seed to use every time, ensuring runs are reproducible
seed = 40

def f(x):
    return jnp.sin(x)+jnp.cos(x**2)

def random_data(N):
  rng = np.random.default_rng(seed)
  x_data = rng.uniform(-np.pi, np.pi, size=N)

  x_data = jnp.asarray(x_data)

  func_val = f(x_data)

  noise = 0.1 * rng.normal(0, 1, size=N)

  noise = jnp.asarray(noise)

  y_data = func_val + noise
  return(x_data, y_data)


x_data, y_data = random_data(N)

#2. Parameters

def init_params(sizes):
  out = []
  key = jax.random.PRNGKey(seed)
  for i in range(len(sizes) - 1):
    key, sub = jax.random.split(key)
    scale = jnp.sqrt(1.0 / sizes[i])
    W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
    b = jnp.zeros(sizes[i + 1])
    out.append((W, b))
  return out

LAYERS = [1, 16, 1]
params = init_params(LAYERS)

#3. Forward pass

def mlp(params, inp):
    for i in range(len(params)-1):
        W, b = params[i]
        inp = jnp.dot(inp, W) + b
        inp = jnp.tanh(inp)
    W, b = params[-1]
    inp = jnp.dot(inp, W) + b
    return inp


def predict_one(x, params):

    inp = jnp.array([x])

    return mlp(params, inp)[0]


#4. Many points at once

predict = jax.vmap(predict_one, in_axes=(0, None))

#5. Loss

def loss(params):
   cost = jnp.mean((predict(x_data, params) - y_data)**2)
   return cost

#6. Gradients

loss_and_grad = jax.value_and_grad(loss)

loss_value, gradients = loss_and_grad(params)

#7. Optimiser

opt = optax.adam(learning_rate = 0.01)
opt_state = opt.init(params)


#8. Training step

@jax.jit   # Compiles function on the first call

def step(params, opt_state):

    loss_value, grads = loss_and_grad(params)

    updates, opt_state = opt.update(grads, opt_state)

    params = optax.apply_updates(params, updates)

    return params, opt_state, loss_value

losses = []
steps = []


for i in range(1, 5001):

    params, opt_state, loss_value = step(params, opt_state)



    if i % 200 == 0:
        losses.append(float(loss_value))
        steps.append(i)
        print("Step:", i, "Loss:", loss_value)

#9. Plot




# Create x values for plotting
x_plot = jnp.linspace(-jnp.pi, jnp.pi, 200)

# Calculate the true function and compare to neural network predictions
y_true = jnp.sin(x_plot)+jnp.cos(x_plot**2)
y_pred = predict(x_plot, params)

# First graph: Function approximation
plt.figure(figsize=(8, 5))

plt.scatter(x_plot, y_true, label="True Function")

plt.plot(x_plot, y_pred, label="Neural network prediction", color="red")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Function Approximation")
plt.legend()
plt.show()


# Second graph: Training loss over time
plt.figure(figsize=(8, 5))

plt.plot(steps, losses)



plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.yscale("log")
plt.show()
