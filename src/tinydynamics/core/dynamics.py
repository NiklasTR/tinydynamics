from tinydynamics.core.state import State
from tinydynamics.core.potentials import general_well_potential
from tinydynamics.core.state import simple_lambda

import jax
import jax.numpy as jnp


def newtonian_dynamics(
    state0: State,
    dt: float,
    n_steps: int,
    mass: float,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    # lambda to skip the passing of a, b, c to the gradient function
    grad_potential = jax.grad(lambda x: general_well_potential(x, a, b, c))

    # one step of the dynamics
    def step(state: State, _: None) -> State:
        v = state.v + dt * -grad_potential(state.x) / mass
        x = state.x + dt * v
        l = simple_lambda(x)
        carry = State(t=state.t + dt, x=x, v=v, l=l, key=state.key) # key pass through
        record = carry # carry is the markovian state, record is what is returned
        return carry, record
    
    # jax.lax.scan to run the dynamics for n_steps steps
    _, records = jax.lax.scan(step, state0, None, n_steps)
    return records


def langevin_dynamics(
    state0: State,
    dt: float,
    n_steps: int,
    mass: float,
    gamma: float,
    temperature: float,
    kB: float = 1.0,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    grad_potential = jax.grad(lambda x: general_well_potential(x, a, b, c))
    sigma = jnp.sqrt(2.0 * gamma * kB * temperature) / mass

    # one step of the dynamics
    def step(state: State, _: None) -> State:
        key, subkey = jax.random.split(state.key) # need subkey for collision term
        xi = jax.random.normal(subkey, shape=state.x.shape)

        v = state.v + dt * -grad_potential(state.x) / mass # potential term
        v -= gamma * state.v / mass # damping term
        v += sigma * xi * jnp.sqrt(dt) # termal term

        x = state.x + dt * v # euler-maruyama step
        l = simple_lambda(x)
        carry = State(t=state.t + dt, x=x, v=v, l=l, key=key)
        record = carry # carry is the markovian state, record is what is returned
        return carry, record
    
    _, records = jax.lax.scan(step, state0, None, n_steps)
    return records
