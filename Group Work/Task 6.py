# 6 - Network potential
# Create initial weights and biases
def init_params(sizes):
    out = []
    key = jax.random.PRNGKey(32)
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(1.0 / sizes[i])
        W = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
        b = jnp.zeros(sizes[i + 1])
        out.append((W, b))
    return out

LAYERS = [2, 24, 24, 1]
p_V = init_params(LAYERS)

# Calculates the output based on weights and biases
def mlp(parameters, inp):
    for i in range(len(parameters)):
        W, b = parameters[i]
        inp = jnp.dot(inp, W) + b
        if i < len(parameters)-1:
            inp = jnp.tanh(inp)      
    return inp

# Calculates the potential at (x,y) based on the weights and biases
def V_net(p_V, x, y):
    inp = jnp.stack([x / L, y / L])
    raw = mlp(p_V, inp)[0]
    return jax.nn.softplus(raw - 2.0)

dVnetdx = jax.grad(V_net, argnums = 1)
dVnetdy = jax.grad(V_net, argnums = 2)
def force_net(p_V,x,y):
    return jnp.array([-dVnetdx(p_V,x,y), -dVnetdy(p_V,x,y)])
