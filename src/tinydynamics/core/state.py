from typing import NamedTuple
import jax.numpy as jnp
import jax

class State(NamedTuple):
    t: jnp.ndarray # (1)
    x: jnp.ndarray # (D)
    v: jnp.ndarray # (D)
    l: bool # (1) binary
    key: jax.Array # (2) PRNGKey for the dynamics


def simple_lambda(x: jnp.ndarray, threshold: float = 2.0) -> bool: # we define lambda as a binary variable
    return jnp.min(x) > threshold # the smallest dimension of the state must be greater than the threshold


def initialize_state(D: int, key: jax.Array) -> State:
    key, kx, kv = jax.random.split(key, 3)

    x = jax.random.uniform(kx, shape=(D), minval=-3.0, maxval=3.0)
    v = jax.random.uniform(kv, shape=(D), minval=-0.1, maxval=0.1)
    l = simple_lambda(x) # starting state
    
    return State(t=jnp.asarray(0.0, dtype=x.dtype), x=x, v=v, l=l, key=key)

