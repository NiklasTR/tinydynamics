from tinydynamics.core.state import State

import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_trajectory(trajectories: State | list[State], filename: str = "trajectory.png") -> None:
    """Plot 2D trajectory in phase space with lambda state over time.
    
    Args:
        trajectories: Single State object or list of State objects with trajectory data
        filename: Path to save figure (default: trajectory.png)
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
                    l=trajectories.l[i],
                    key=trajectories.key[i]
                )
                for i in range(B)
            ]
        else:
            trajectories = [trajectories]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), 
                                    gridspec_kw={'height_ratios': [8, 1]})
    
    from matplotlib.collections import LineCollection
    
    n_traj = len(trajectories)
    base_colors = plt.cm.tab10(jnp.linspace(0, 1, n_traj))
    
    for idx, traj in enumerate(trajectories):
        n_points = len(traj.t)
        base_color = base_colors[idx][:3]
        
        # Left plot: Phase space trajectory
        points = jnp.column_stack([traj.x[:, 0], traj.x[:, 1]])
        segments = jnp.stack([points[:-1], points[1:]], axis=1)
        
        alphas = jnp.linspace(0.1, 0.4, n_points - 1)
        colors = [
            [float(base_color[0]), float(base_color[1]), float(base_color[2]), float(a)] 
            for a in alphas
        ]
        
        lc = LineCollection(segments, colors=colors, linewidth=0.5)
        ax1.add_collection(lc)
        
        ax1.plot(traj.x[0, 0], traj.x[0, 1], 'o', color=base_color, 
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, 
                zorder=10, label=f'Traj {idx+1}' if n_traj > 1 else None)
        ax1.plot(traj.x[-1, 0], traj.x[-1, 1], 's', color=base_color, 
                markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        # Right plot: Lambda state over time
        ax2.plot(traj.t, traj.l.astype(float), color=base_color, linewidth=1.5, alpha=0.7)
    
    # Configure left plot (phase space)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Phase Space Trajectories', fontsize=14)
    ax1.grid(alpha=0.3)
    
    if n_traj > 1:
        ax1.legend(loc='upper right', fontsize=10)
    
    # Configure right plot (lambda state)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Lambda State', fontsize=12)
    ax2.set_title('Lambda State Over Time', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

