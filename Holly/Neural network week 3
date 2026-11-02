import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import optax

# generate data points
x_data = jnp.linspace(0, 2*jnp.pi, 50)
# y_data = jnp.sin(x_data) + jnp.sin(2*x_data)
y_data = jnp.sin(1.5*x_data) + 0.3*jnp.cos(4*x_data)

# create a neural network, copy from notes
def initial_parameters(sizes):
    parameters = []
    key = jax.random.PRNGKey(32)
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(1.0 / sizes[i])
        W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
        b = jnp.zeros(sizes[i + 1])
        parameters.append((W, b))
    return parameters

layers = [1, 16, 1]
parameters = initial_parameters(layers)
iterations = 10000

opt = optax.adam(learning_rate = 0.01)
opt_state = opt.init(parameters)

# calculate (x,y hat) pairs
def network(parameters, x):
    # rescale x
    x_rescaled = jnp.array([(x/jnp.pi) -1])
    # calculate 16 neuron values
    W1, b1 = parameters[0]
    n = 1 / (1 + jnp.exp(-(jnp.dot(x_rescaled, W1) + b1)))
    # produce y_hat
    W2, b2 = parameters[1]
    y_hat = jnp.dot(n, W2)[0] + b2[0]
    return y_hat

# adjust weights and biases
# calculate loss
def loss(parameters):
    loss = 0
    for i in range(len(x_data)):
        y_hat = network(parameters, x_data[i])
        loss = loss + (y_hat - y_data[i])**2 
    return loss/len(x_data)

# calculate gradient of loss
gradient_loss = jax.grad(loss)

# update weights and biases
def training(a, _):
    parameters, opt_state = a
    gradients = gradient_loss(parameters)
    updates, opt_state = opt.update(gradients, opt_state, parameters)
    new_parameters = optax.apply_updates(parameters, updates)
    loss1 = loss(new_parameters)
    return (new_parameters, opt_state), loss1

initial_parameters_loss = loss(parameters)
(a, loss1) = jax.lax.scan(training, (parameters, opt_state), length = iterations)
parameters, opt_state = a

# plots
x_plot = jnp.linspace(0, 2*jnp.pi, 200)
y_plot = []
for x in x_plot:
    y_plot.append(network(parameters, x))

#true_function = (jnp.sin(x_plot) + jnp.sin(2 * x_plot))
true_function = jnp.sin(1.5*x_plot) + 0.3*jnp.cos(4*x_plot)

plt.figure()
plt.scatter(x_data, y_data, label = "Data", color = "black", linewidth = 2)
plt.plot(x_plot, true_function, label = "True function",  color = "black", linewidth = 2)
plt.plot(x_plot, y_plot, label = "Neural network",  color = "hotpink", linestyle = '--', linewidth = 2)
plt.legend()
plt.show()
