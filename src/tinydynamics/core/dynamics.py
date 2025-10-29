from tinydynamics.core.state import State
from tinydynamics.core.environment import general_well_potential
from tinydynamics.core.state import simple_cv
from tinydynamics.core.teacher import constant_pull

import jax
import jax.numpy as jnp


def newtonian_dynamics(
    state0: State,
    dt: float,
    n_steps: int,
    mass: float,
    #policy: Policy,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    # lambda to skip the passing of a, b, c to the gradient function
    def system_potential(x: jnp.ndarray, t: float) -> jnp.ndarray:
        return general_well_potential(x, a, b, c) + constant_pull(x, t, n_steps) #+ policy_forward(x, policy)
    
    grad_Ux = jax.grad(system_potential, argnums=0) # force
    grad_Ut = jax.grad(system_potential, argnums=1) # work

    # one step of the dynamics
    def step(state: State, _: None) -> State:
        v = state.v + dt * -grad_Ux(state.x, state.t) / mass
        x = state.x + dt * v
        cv = simple_cv(x)
        w = state.w + grad_Ut(state.x, state.t) * dt # cumulative work
        carry = State(t=state.t + dt, x=x, v=v, cv=cv, w=w, key=state.key) # key pass through
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
    #policy: Policy,
    kB: float = 1.0,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    def system_potential(x: jnp.ndarray, t: float) -> jnp.ndarray:
        return general_well_potential(x, a, b, c) + constant_pull(x, t, n_steps) #+ policy_forward(x, policy)

    grad_Ux = jax.grad(system_potential, argnums=0) # force
    grad_Ut = jax.grad(system_potential, argnums=1) # work
    sigma = jnp.sqrt(2.0 * gamma * kB * temperature) / mass

    # one step of the dynamics
    def step(state: State, _: None) -> State:
        key, subkey = jax.random.split(state.key) # need subkey for collision term
        xi = jax.random.normal(subkey, shape=state.x.shape)

        v = state.v + dt * -grad_Ux(state.x, state.t) / mass # potential term
        v -= gamma * state.v / mass # damping term
        v += sigma * xi * jnp.sqrt(dt) # termal term

        x = state.x + dt * v # euler-maruyama step
        cv = simple_cv(x)
        w = state.w + grad_Ut(state.x, state.t) * dt # cumulative work
        carry = State(t=state.t + dt, x=x, v=v, cv=cv, w=w, key=key)
        record = carry # carry is the markovian state, record is what is returned
        return carry, record
    
    _, records = jax.lax.scan(step, state0, None, n_steps)
    return records
