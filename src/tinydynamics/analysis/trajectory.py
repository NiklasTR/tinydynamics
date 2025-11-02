from tinydynamics.core.state import State
import jax
import jax.numpy as jnp
from scipy.optimize import curve_fit


def velocity_autocorrelation(
    trajectory: State,
    max_lag: int | None = None,
    normalize: bool = True,
    downsample: int = 10
) -> jnp.ndarray:
    """Compute velocity autocorrelation function from trajectory.
    
    The VACF measures how velocity correlations decay over time:
    C_v(tau) = <v(t) · v(t+tau)>
    
    Signature distinguishes damping regimes:
    - Underdamped: oscillations before decay
    - Overdamped: monotonic exponential decay
    
    Args:
        trajectory: State object with velocity data (n_steps, D)
        max_lag: Maximum lag to compute (default: n_steps // 2)
        normalize: If True, normalize by C_v(0) so VACF starts at 1.0
        downsample: Compute only every Nth lag to reduce cost (default: 10)
    
    Returns:
        Array of shape (max_lag // downsample,) with VACF values
    """
    velocities = trajectory.v
    n_steps = velocities.shape[0]
    
    if max_lag is None:
        max_lag = n_steps // 2
    
    max_lag = min(max_lag, n_steps - 1)
    
    # Downsample lag indices to reduce computation
    lag_indices = jnp.arange(0, max_lag, downsample)
    
    def compute_single_correlation(lag: int) -> float:
        n_pairs = n_steps - lag
        v_t = velocities[:n_pairs]
        v_t_plus_lag = velocities[lag:lag + n_pairs]
        return jnp.mean(jnp.sum(v_t * v_t_plus_lag, axis=-1))
    
    vacf = jnp.array([compute_single_correlation(int(lag)) for lag in lag_indices])
    
    if normalize:
        vacf = vacf / vacf[0]
    
    return vacf


def fit_vacf_decay_time(lag_times: jnp.ndarray, vacf: jnp.ndarray) -> float | None:
    """Fit exponential decay to VACF and extract decay time constant tau.
    
    Fits: C_v(t) = exp(-t/tau)
    
    Args:
        lag_times: Array of lag times
        vacf: Normalized VACF values (should start at ~1.0)
    
    Returns:
        tau: Decay time constant, or None if fit fails
    """
    try:
        # Only fit positive values (where VACF > 0) for log-linear fit
        mask = vacf > 0.01  # Avoid numerical issues near zero
        if jnp.sum(mask) < 3:
            return None
        
        t_fit = lag_times[mask]
        c_fit = vacf[mask]
        
        # Exponential decay: C(t) = exp(-t/tau)
        def exp_decay(t, tau):
            return jnp.exp(-t / tau)
        
        # Initial guess: tau ~ time when VACF drops to 1/e
        initial_tau = float(jnp.max(t_fit)) / 3.0
        
        popt, _ = curve_fit(
            exp_decay, 
            t_fit, 
            c_fit, 
            p0=[initial_tau],
            bounds=(0, jnp.inf),
            maxfev=1000
        )
        
        tau = float(popt[0])
        return tau
    except Exception:
        return None

