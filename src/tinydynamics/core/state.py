from typing import NamedTuple
import jax.numpy as jnp
import jax

class State(NamedTuple):
    t: jnp.ndarray # (1)
    x: jnp.ndarray # (D)
    v: jnp.ndarray # (D)
    cv: float # (1) continuous variable
    w: float # (1) work
    key: jax.Array # (2) PRNGKey for the dynamics


def simple_cv(x: jnp.ndarray, threshold: float = 2.0, mu: float = 3.0, sigma: float = 1.0) -> float: # we define lambda as a binary variable
    on_target = jnp.min(x) > threshold # the smallest dimension of the state must be greater than the threshold
    rmsd = jnp.where(on_target, 0.0, jnp.sqrt(jnp.mean((x - threshold) ** 2)))  # RMSD to threshold point if not on target
    z = (rmsd - mu) / sigma
    cv = 1.0 / (1.0 + jnp.exp(-z))
    return cv

def initialize_state(D: int, key: jax.Array) -> State:
    key, kx, kv = jax.random.split(key, 3)

    x = jax.random.uniform(kx, shape=(D), minval=-6.0, maxval=-1.0) # lambda = False
    v = jax.random.uniform(kv, shape=(D), minval=-0.1, maxval=0.1)
    cv = simple_cv(x) # starting state
    w = 0.0 # starting work
    
    return State(t=jnp.asarray(0.0, dtype=x.dtype), x=x, v=v, cv=cv, w=w, key=key)

