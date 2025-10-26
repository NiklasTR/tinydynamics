from typing import NamedTuple
import jax.numpy as jnp
import jax

class State(NamedTuple):
    t: jnp.ndarray # () scalar
    x: jnp.ndarray # (D)
    v: jnp.ndarray # (D)
    key: jax.Array # (2) PRNGKey for the dynamics


def initialize_state(D: int, key: jax.Array) -> State:
    key, kx, kv = jax.random.split(key, 3)
    x = jax.random.uniform(kx, shape=(D), minval=-3.0, maxval=3.0)
    v = jax.random.uniform(kv, shape=(D), minval=-0.1, maxval=0.1)
    return State(t=jnp.asarray(0.0, dtype=x.dtype), x=x, v=v, key=key)
