from tinydynamics.core.state import State

import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_trajectory(trajectories: State | list[State], filename: str | None = None) -> None:
    """Plot 2D trajectory in phase space.
    
    Args:
        trajectories: Single State object or list of State objects with trajectory data
        filename: Optional path to save figure
    """
    if not isinstance(trajectories, list):
        # Check if it's a batched State (has B dimension)
        if trajectories.x.ndim == 3:  # (B, n_steps, D)
            # Unbatch: convert to list of individual trajectories
            B = trajectories.x.shape[0]
            trajectories = [
                State(
                    t=trajectories.t[i],
                    x=trajectories.x[i],
                    v=trajectories.v[i],
                    key=trajectories.key[i]
                )
                for i in range(B)
            ]
        else:
            trajectories = [trajectories]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    
    n_traj = len(trajectories)
    base_colors = plt.cm.tab10(jnp.linspace(0, 1, n_traj))
    
    for idx, traj in enumerate(trajectories):
        n_points = len(traj.t)
        base_color = base_colors[idx][:3]
        
        points = jnp.column_stack([traj.x[:, 0], traj.x[:, 1]])
        segments = jnp.stack([points[:-1], points[1:]], axis=1)
        
        alphas = jnp.linspace(0.1, 0.4, n_points - 1)
        colors = [
            [float(base_color[0]), float(base_color[1]), float(base_color[2]), float(a)] 
            for a in alphas
        ]
        
        lc = LineCollection(segments, colors=colors, linewidth=0.75)
        ax.add_collection(lc)
        
        ax.plot(traj.x[0, 0], traj.x[0, 1], 'o', color=base_color, 
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, 
                zorder=10, label=f'Traj {idx+1}' if n_traj > 1 else None)
        ax.plot(traj.x[-1, 0], traj.x[-1, 1], 's', color=base_color, 
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Phase Space Trajectories', fontsize=14)
    ax.grid(alpha=0.3)
    
    if n_traj > 1:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()

