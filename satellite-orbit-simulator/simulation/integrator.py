from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from simulation.physics import state_derivative


def propagate_orbit(
    initial_state: np.ndarray,
    duration: float,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a spacecraft state forward in time using a numerical ODE solver.

    The orbit is integrated under Earth-centered gravity using SciPy's solve_ivp with
    the RK45 method. The state vector is ordered as [x, y, z, vx, vy, vz], with
    position in kilometers, velocity in km/s, and time in seconds.

    Args:
        initial_state: Six-element state vector [x, y, z, vx, vy, vz].
        duration: Total integration time in seconds. Must be positive.
        time_step: Desired output sampling interval in seconds. Must be positive.

    Returns:
        A tuple (times, states), where times is a 1D array of output times and states
        is a 2D array of shape (len(times), 6) containing the propagated state at each
        time.

    Raises:
        ValueError: If the state vector or time inputs are invalid.
        RuntimeError: If the numerical integration fails.
    """
    state_array = np.asarray(initial_state, dtype=float)

    if state_array.shape != (6,):
        raise ValueError("initial_state must be a six-element NumPy array [x, y, z, vx, vy, vz].")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("initial_state contains non-finite values.")

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive finite number in seconds.")

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be a positive finite number in seconds.")

    if np.linalg.norm(state_array[:3]) <= 0.0:
        raise ValueError("initial position magnitude must be greater than zero.")

    num_steps = max(1, int(np.ceil(duration / time_step)))
    output_times = np.linspace(0.0, duration, num_steps + 1)

    solution = solve_ivp(
        state_derivative,
        (0.0, duration),
        state_array,
        method="RK45",
        t_eval=output_times,
        rtol=1e-9,
        atol=1e-12,
    )

    if not solution.success:
        raise RuntimeError(f"Orbit propagation failed: {solution.message}")

    if len(solution.t) != len(output_times):
        raise RuntimeError(
            "Orbit propagation did not return all requested output samples and did not reach the requested final time. "
            f"Expected {len(output_times)} samples but solver returned {len(solution.t)}."
        )

    return solution.t, solution.y.T
