import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import optax
import numpy as np

# 1. Data
x_data = jnp.linspace(-jnp.pi, jnp.pi, 50)
rng = np.random.default_rng(100)
y_data = jnp.sin(x_data) + jnp.cos(x_data**2) + 0.1*jnp.asarray(rng.uniform(low = -1, high = 1, size = 50))

# 2. Parameters, from section 5.1
def initial_parameters(sizes):
    out = []
    key = jax.random.PRNGKey(32)
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(1.0 / sizes[i])
        W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
        b = jnp.zeros(sizes[i + 1])
        out.append((W, b))
    return out

layers = [1, 32, 1]
parameters = initial_parameters(layers)

for W, b in parameters:
    print("W shape:", W.shape)
    print("b shape:", b.shape)
  
# Why should the initial W be random rather than zero?
# If the weights were zero then they would all be updated the same from the gradient which cancels out the benefit of having 16 neurons.

# 3. Forward pass
def mlp(parameters, inp):
    for i in range(len(parameters)):
        W, b = parameters[i]
        inp = jnp.dot(inp, W) + b
        if i < len(parameters)-1:
            inp = jnp.tanh(inp)      
    return inp

# tanh is not applied to the last layer because the output does not need to be in the range -1 to 1.

def predict_one(x, parameters):
    x = jnp.array([x])
    return mlp(parameters, x)[0]

# 4. Many points at once
predict = jax.vmap(predict_one, in_axes=(0, None))

print("predict shape:", predict(x_data, parameters).shape)
print("y_data shape:", y_data.shape)

# 5. Loss
def loss(parameters):
    return jnp.mean((predict(x_data, parameters) - y_data)**2)

# 6. Gradients
loss_and_grad = jax.value_and_grad(loss)

loss_value, gradients = loss_and_grad(parameters)
for dW, db in gradients:
    print("dW shape:", dW.shape)
    print("db shape:", db.shape)

# 7. Optimiser
opt = optax.adam(learning_rate = 0.01)
opt_state = opt.init(parameters)

# If the learning rate is too small, it does not reach a good solution in the number of iterations. But if the number of iterations is bigger, then the soltion is better.
# If the learning rate is too large, it can miss a minimum in the loss function and become a inaccurate.

# 8. Training step
@jax.jit
def step(parameters, opt_state):
    loss, gradients = loss_and_grad(parameters)
    updates, opt_state = opt.update(gradients, opt_state)
    parameters = optax.apply_updates(parameters, updates)
    return parameters, opt_state, loss

iterations = 3001
current_loss = []
for i in range(iterations):
    parameters, opt_state, loss = step(parameters, opt_state)
    current_loss.append(float(loss))
    if i%200 == 0:
      print("Iteration", i, "Loss =", float(loss))

# 9. Plot
x_plot = jnp.linspace(-jnp.pi, jnp.pi, 200)
y_plot = predict(x_plot, parameters)
true_function = jnp.sin(x_plot) + jnp.cos(x_plot**2)

plt.figure()
plt.plot(x_plot, true_function, label = "True function", color = "red", linewidth = 2)
plt.plot(x_data, y_data, label = "Data", color = "black", linewidth = 2)
plt.scatter(x_data, y_data, label = "Data points", color = "black", linewidth = 1)
plt.plot(x_plot, y_plot, label = "Neural network",  color = "hotpink", linestyle = '--', linewidth = 2)
plt.legend()

x_values = range(1, iterations+1)
plt.figure()
plt.semilogy(x_values, current_loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# 10. 
# Increasing the number of layers or the number of neurons in each layer reduces the loss much faster, since the network has more parameters so can adjust and fit the data better.
# However, if there are too many layers or neurons the network learns the random noise as well as the data points, so it becomes inaccurate of the true function.
# This can be seen my increasing the noise in the data points.
