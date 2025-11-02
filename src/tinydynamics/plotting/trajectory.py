from tinydynamics.core.state import State
from tinydynamics.analysis import velocity_autocorrelation, fit_vacf_decay_time

import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_trajectory(trajectories: State | list[State], filename: str = "trajectory.png", plot_vacf: bool = False) -> None:
    """Plot 2D trajectory in phase space with lambda state, work, and optionally VACF.
    
    Args:
        trajectories: Single State object or list of State objects with trajectory data
        filename: Path to save figure (default: trajectory.png)
        plot_vacf: If True, compute and plot velocity autocorrelation (expensive, default: False)
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
                    cv=trajectories.cv[i],
                    w=trajectories.w[i],
                    key=trajectories.key[i]
                )
                for i in range(B)
            ]
        else:
            trajectories = [trajectories]
    
    if plot_vacf:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), 
                                        gridspec_kw={'height_ratios': [8, 1, 1, 2]})
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13), 
                                        gridspec_kw={'height_ratios': [8, 1, 1]})
    
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
        
        # Second plot: Lambda state over time
        ax2.plot(traj.t, traj.cv.astype(float), color=base_color, linewidth=1.5, alpha=0.7)
        
        # Third plot: Work over time
        ax3.plot(traj.t, traj.w.astype(float), color=base_color, linewidth=1.5, alpha=0.7)
        
        # Fourth plot: Velocity autocorrelation (optional)
        if plot_vacf:
            dt = traj.t[1] - traj.t[0]
            max_lag = min(int(100 / dt), len(traj.t) // 2)  # Ensure we reach lag time = 100
            downsample = 10  # Compute every 10th lag for efficiency
            vacf = velocity_autocorrelation(traj, max_lag=max_lag, normalize=True, downsample=downsample)
            # Lag indices that were actually computed (downsampled)
            lag_indices = jnp.arange(0, max_lag, downsample)[:len(vacf)]
            lag_times_all = lag_indices * dt
            lag_times = lag_times_all[1:]  # Start from index 1 to avoid log(0) in plot
            ax4.plot(lag_times, vacf[1:].astype(float), color=base_color, linewidth=1.5, alpha=0.7)
            
            # Fit decay time and display
            tau = fit_vacf_decay_time(lag_times_all, vacf)
            if tau is not None:
                # Add text box with tau value
                textstr = f'τ = {tau:.3f}'
                props = dict(boxstyle='round', facecolor=base_color, alpha=0.3, edgecolor=base_color)
                ax4.text(0.95, 0.95, textstr, transform=ax4.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right', bbox=props)
    
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
    
    # Configure second plot (lambda state)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Lambda State', fontsize=12)
    ax2.set_title('Lambda State Over Time', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Configure third plot (work)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Work', fontsize=12)
    ax3.set_title('Work Over Time', fontsize=14)
    ax3.grid(alpha=0.3)
    
    # Configure fourth plot (VACF) - only if enabled
    if plot_vacf:
        ax4.set_xlabel('Lag Time', fontsize=12)
        ax4.set_ylabel('Normalized VACF', fontsize=12)
        ax4.set_title('Velocity Autocorrelation Function', fontsize=14)
        ax4.set_xscale('log')
        ax4.set_xlim(1e-2, 1e2)
        ax4.grid(alpha=0.3, which='both')
        ax4.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

