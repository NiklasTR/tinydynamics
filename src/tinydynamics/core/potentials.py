import jax.numpy as jnp

def general_well_potential(x: jnp.ndarray, a: float, b: float, c: float) -> jnp.ndarray:
    """
    single_well_potential: a=1.0, b=0.0, c=0.0
    symmetric_double_well_potential: a=1.0, b=3.0, c=0.0
    asymmetric_double_well_potential: a=1.0, b=3.0, c=10.0
    """
    return jnp.sum(a * (x**2 - b**2)**2 + c * x)
