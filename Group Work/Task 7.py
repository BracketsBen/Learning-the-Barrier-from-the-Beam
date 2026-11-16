# 7 - Predictions
# Predict the exit position and vertical velocity for one particle
def predict_exit(y0, vy0, p_V):
    
    def rhs_fn_net(state):
        x, y, vx, vy = state
        return jnp.array([vx, vy, -dVnetdx(p_V,x,y), -dVnetdy(p_V,x,y)])
    
    trajectory = integrate_traj(y0, vy0, rhs_fn_net)
    return exit_state_from_traj(trajectory)

# Predict the exit data for all particles
predict_exitvalues_allparticles = jax.vmap(predict_exit, in_axes=(0, 0, None))
predicted_exitvalues_allparticles = predict_exitvalues_allparticles(y0_values, vy0_values, p_V)

print(predicted_exitvalues_allparticles.shape)
print(true_exit_values.shape)
# These are both (100, 2)
