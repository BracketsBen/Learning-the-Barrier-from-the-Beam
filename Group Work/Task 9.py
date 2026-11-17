#9. Optimiser

schedule = optax.exponential_decay(
    init_value=1e-2,
    transition_steps=600,
    decay_rate=1e-2
)

clip = optax.clip_by_global_norm(0.1)

opt = optax.chain(clip, optax.adam(schedule))
  
opt_state = opt.init(params)
