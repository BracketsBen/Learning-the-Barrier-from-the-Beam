
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#pseudo random key
key = jax.random.PRNGKey(42)


#original function
def f(x):
    return jnp.sin(1.5 * x) + 0.3 * jnp.cos(4 * x)


#number of data points, use the key for x_data and the fct for y_data
N = 100
key, subkey = jax.random.split(key)

x_data = jax.random.uniform(
    subkey, (N, 1), minval=-3.0, maxval=3.0
)

y_data = f(x_data)


#splits of data- 70/15/15
key, subkey = jax.random.split(key)

indices = jax.random.permutation(subkey, N)

train_end = int(0.7 * N)
val_end = int(0.85 * N)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

x_train = x_data[train_idx]
y_train = y_data[train_idx]

x_val = x_data[val_idx]
y_val = y_data[val_idx]

x_test = x_data[test_idx]
y_test = y_data[test_idx]


#weights and biases
key, k1, k2 = jax.random.split(key, 3)

W1 = jax.random.normal(k1, (1, 16)) * 0.5
b1 = jnp.zeros(16)

W2 = jax.random.normal(k2, (16, 1)) * 0.5
b2 = jnp.zeros(1)

params = (W1, b1, W2, b2)


#allows for non-linearity
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


#weight updating
def predict(params, x):
    W1, b1, W2, b2 = params

    hidden = sigmoid(x @ W1 + b1)
    output = hidden @ W2 + b2

    return output


#loss function
def loss_fn(params, x, y):
    predictions = predict(params, x)
    return jnp.mean((predictions - y) ** 2)


#original model parameters- using the m and v for adam, the mean past weights and exponentiated past weights
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m = jax.tree.map(jnp.zeros_like, params)
v = jax.tree.map(jnp.zeros_like, params)


#the model- used gAI to write the majority- calculate gradients, update adam weights, adjust weights to reduce error
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

    m_hat = jax.tree.map(
        lambda a: a / (1 - beta1**t), m
    )

    v_hat = jax.tree.map(
        lambda a: a / (1 - beta2**t), v
    )

    params = jax.tree.map(
        lambda p, a, b: p - learning_rate * a
        / (jnp.sqrt(b) + epsilon),
        params, m_hat, v_hat
    )

    return params, m, v, loss


#5000 passes through for thorough updating
n_epochs = 5000

for epoch in range(1, n_epochs + 1):

    params, m, v, loss = train_step(
        params, m, v, jnp.array(epoch),
        x_train, y_train
    )

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")


#define the losses of the 3 splits
train_loss = loss_fn(params, x_train, y_train)
val_loss = loss_fn(params, x_val, y_val)
test_loss = loss_fn(params, x_test, y_test)

print(f"Training loss: {train_loss:.6f}")
print(f"Validation loss: {val_loss:.6f}")
print(f"Test loss: {test_loss:.6f}")

#plotting x values and refitting
x_plot = jnp.linspace(-3, 3, 500).reshape(-1, 1)

y_true = f(x_plot)
y_pred = predict(params, x_plot)


#full plotting
plt.figure(figsize=(10, 6))

plt.scatter(
    x_train[:, 0], y_train[:, 0],
    label="Training data"
)

plt.scatter(
    x_test[:, 0], y_test[:, 0],
    label="Test data"
)

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
plt.title("Neural Network Curve Fitting")
plt.legend()
plt.grid(True)
plt.show()

#originally used a 70/15/15 split, however after testing a 50/25/25 split, the loss of the test data drops significantly- from 0.0144 to 0.0109
#need to ask whether testing the splits is helpful or not, and what is the best choice for number of neurons- I'm not quite sure what the difference between
#say 16, 32, 64 etc
#I presume that more neurons could introduce overfitting- as the loss for training, validating and testing is low it seems 16 is a good baseline- I will have a
#play about with this however and see, but I'd rather understand it properly
