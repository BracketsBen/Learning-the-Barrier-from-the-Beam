
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Invent data
key = jax.random.PRNGKey(42)

def f(x):
    return jnp.sin(1.5 * x) + 0.3 * jnp.cos(4 * x)

N = 100
key, subkey = jax.random.split(key)

x_data = jax.random.uniform(
    subkey, (N, 1), minval=-3.0, maxval=3.0
)
y_data = f(x_data)

# Organise data
idx = jnp.argsort(x_data[:, 0])
x_data = x_data[idx]
y_data = y_data[idx]

# Make NN
# One input, 16 hidden neurons, one output
key, k1, k2 = jax.random.split(key, 3)

W1 = jax.random.normal(k1, (1, 16)) * 0.5
b1 = jnp.zeros(16)
W2 = jax.random.normal(k2, (16, 1)) * 0.5
b2 = jnp.zeros(1)

params = (W1, b1, W2, b2)

def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))

def predict(params, x):
    W1, b1, W2, b2 = params

    hidden = sigmoid(x @ W1 + b1)
    output = hidden @ W2 + b2

    return output

# Loss meme
def loss_fn(params, x, y):
    predictions = predict(params, x)
    return jnp.mean((predictions - y) ** 2)

# training
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m = jax.tree.map(jnp.zeros_like, params)
v = jax.tree.map(jnp.zeros_like, params)

@jax.jit
def train_step(params, m, v, t, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

    m = jax.tree.map(
        lambda old, g: beta1 * old + (1 - beta1) * g,
        m, grads
    )
    v = jax.tree.map(
        lambda old, g: beta2 * old + (1 - beta2) * g**2,
        v, grads
    )

    m_hat = jax.tree.map(lambda a: a / (1 - beta1**t), m)
    v_hat = jax.tree.map(lambda a: a / (1 - beta2**t), v)

    params = jax.tree.map(
        lambda p, a, b: p - learning_rate * a
        / (jnp.sqrt(b) + epsilon),
        params, m_hat, v_hat
    )

    return params, m, v, loss

n_epochs = 5000

for epoch in range(1, n_epochs + 1):
    params, m, v, loss = train_step(
        params, m, v, jnp.array(epoch),
        x_data, y_data
    )

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")

# bob ross
x_plot = jnp.linspace(-3, 3, 500).reshape(-1, 1)
y_true = f(x_plot)
y_pred = predict(params, x_plot)

plt.figure(figsize=(10, 6))

#plt.scatter(
#    x_data[:, 0], y_data[:, 0],
#    label="Data", color="salmon"
#)
plt.plot(
    x_plot[:, 0], y_true[:, 0],
    label="True function", color="goldenrod"
)
plt.plot(
    x_plot[:, 0], y_pred[:, 0],
    "--", label="Neural network", color="chartreuse"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural network curve fitting")
plt.legend()
plt.grid(True)
plt.show()
