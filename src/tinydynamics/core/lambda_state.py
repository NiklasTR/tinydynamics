from tinydynamics.core.state import State

import jax.numpy as jnp


def get_lambda(state: State, threshold: float) -> jnp.ndarray:
    return jnp.min(state.x, axis=-1) > threshold

