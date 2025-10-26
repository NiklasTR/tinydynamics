from typing import NamedTuple
import jax
import jax.numpy as jnp


class Policy(NamedTuple):
    params: tuple[dict, ...] # variadic tuple type

def initialize_policy(key: jax.Array, layer_sizes: list[int]) -> Policy:
    layers = []
    for i in range(len(layer_sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
        W = jax.random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,))
        layers.append(dict(W=W, b=b))
    return Policy(params=tuple(layers))

def policy_forward(x: jnp.ndarray, policy: Policy) -> jnp.ndarray:
    for layer in policy.params[:-1]:
        x = x @ layer["W"] + layer["b"]
        x = jnp.tanh(x)
    x = x @ policy.params[-1]["W"] + policy.params[-1]["b"]
    return x.squeeze()
