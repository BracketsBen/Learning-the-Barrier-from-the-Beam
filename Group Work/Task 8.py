# 8 - Loss
def loss_fn(p_V):
    predicted_exitvalues_allparticles = predict_exitvalues_allparticles(y0_values, vy0_values, p_V)
    
    predicted_y = predicted_exitvalues_allparticles[:, 0]
    predicted_vy = predicted_exitvalues_allparticles[:, 1]
    
    mse_y = jnp.mean((predicted_y - observed_y_exit_values)**2)
    mse_vy = jnp.mean((predicted_vy - observed_vy_exit_values)**2)

    return 0.5 * ( (mse_y / (noise_y**2)) + (mse_vy / (noise_vy**2)) )

# Calculate the loss and its gradients
loss_value_and_grad = jax.value_and_grad(loss_fn)
loss_value, gradients = loss_value_and_grad(p_V)
