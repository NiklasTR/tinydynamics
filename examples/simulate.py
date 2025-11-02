from tinydynamics.core.state import initialize_state
from tinydynamics.core.dynamics import newtonian_dynamics, langevin_dynamics
from tinydynamics.plotting.trajectory import plot_trajectory

import jax


def simulate_single(
    D: int,
    n_steps: int,
    key: jax.Array,
    dt: float,
    mass: float,
    gamma: float,
    temperature: float,
    kB: float = 1.0,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    state = initialize_state(D, key)
    #records = newtonian_dynamics(state, dt, n_steps, mass, a, b, c)
    records = langevin_dynamics(state, dt, n_steps, mass, gamma, temperature, kB, a, b, c)
    return records


def simulate_batch(
    B: int,
    D: int,
    n_steps: int,
    key: jax.Array,
    dt: float,
    mass: float,
    gamma: float,
    temperature: float,
    kB: float = 1.0,
    a: float = 1.0,
    b: float = 3.0,
    c: float = 10.0,
):
    keys = jax.random.split(key, B)
    states = jax.vmap(initialize_state, in_axes=(None, 0))(D, keys)
    #records = jax.vmap(newtonian_dynamics, in_axes=(0, None, None, None, None, None, None))(states, dt, n_steps, mass, a, b, c)
    records = jax.vmap(langevin_dynamics, in_axes=(0, None, None, None, None, None, None, None, None, None))(states, dt, n_steps, mass, gamma, temperature, kB, a, b, c)
    return records


# Apply JIT compilation with static arguments
simulate_single = jax.jit(simulate_single, static_argnames=("D", "n_steps"))
simulate_batch = jax.jit(simulate_batch, static_argnames=("B", "D", "n_steps"))

if __name__ == "__main__":
    # static arguments
    D = 2
    B = 4
    n_steps = 100_000

    # dynamic arguments
    key = jax.random.PRNGKey(42)
    gamma = 100.0 # as gamma increases we go from inertial to diffusive dynamics
    temperature = 100.0
    kB = 1.0
    dt = 0.01
    mass = 10.0
    a = 1.0
    b = 4.0
    c = 16.0

    # simulate
    records = simulate_batch(B, D, n_steps, key, dt, mass, gamma, temperature, kB, a, b, c)

    # plot
    plot_trajectory(records, plot_vacf=True)