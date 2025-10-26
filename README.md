
tinydynamics/
│
├── core/
│   ├── state.py           # Dataclsass for State (x, v, f, t)
│   ├── forces.py          # potential, damping, thermal
│   ├── integrators.py     # Euler–Maruyama, Verlet, etc.
│   ├── dynamics.py        # total_force, acceleration, simulate()
│
├── plots/
│   ├── visualize.py       # plotting utilities
│
├── examples/
│   ├── brownian.py
│   ├── langevin.py
│   ├── harmonic_oscillator.py
│
└── README.md
