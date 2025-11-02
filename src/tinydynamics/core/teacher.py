import jax.numpy as jnp

def constant_pull(x: jnp.ndarray, t: float, n_steps: int, kappa: float = 0.0) -> jnp.ndarray:

    scaled_t = t / n_steps # from 0 to 1 over time
    target_cv = 1.0 - scaled_t # from 1 to 0 over time
    potential = 0.5 * kappa * (x - target_cv)**2 # pull the x value towards the target_cv value
    
    return jnp.sum(potential)

