import numpy as np

MU_EARTH = 398600.4418  # km^3 / s^2


def gravitational_acceleration(position: np.ndarray) -> np.ndarray:
    """Return the gravitational acceleration vector due to Earth.

    The acceleration is computed from Newtonian gravity using
    a = -MU_EARTH * r / ||r||^3,
    where r is the position vector in kilometers and the result is in km/s^2.

    Args:
        position: A 3D NumPy array representing the spacecraft position in km.

    Returns:
        A NumPy array of shape (3,) containing the acceleration vector in km/s^2.

    Raises:
        TypeError: If the input is not a NumPy array or is not a 3D vector.
        ValueError: If the vector is zero-length or contains non-finite values.
    """
    position_array = np.asarray(position, dtype=float)

    if position_array.shape != (3,):
        raise ValueError("position must be a 3D NumPy vector with exactly 3 elements.")

    if not np.all(np.isfinite(position_array)):
        raise ValueError("position contains non-finite values.")

    radius = np.linalg.norm(position_array)
    if radius == 0.0:
        raise ValueError("position magnitude cannot be zero.")

    acceleration = -MU_EARTH * position_array / (radius ** 3)
    return acceleration.astype(float, copy=False)


def state_derivative(time: float, state: np.ndarray) -> np.ndarray:
    """Return the time derivative of a spacecraft state vector.

    The state is ordered as [x, y, z, vx, vy, vz], where units are km and km/s.
    The derivative returned is [vx, vy, vz, ax, ay, az], with acceleration in km/s^2.

    Args:
        time: Time in seconds. Included to match the ODE signature used by solvers.
        state: Six-element state vector [x, y, z, vx, vy, vz].

    Returns:
        A six-element NumPy array containing the state derivative.

    Raises:
        TypeError: If the state is not a NumPy array-like object.
        ValueError: If the state does not contain six finite elements.
    """
    del time

    state_array = np.asarray(state, dtype=float)

    if state_array.shape != (6,):
        raise ValueError("state must be a six-element vector [x, y, z, vx, vy, vz].")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("state contains non-finite values.")

    position = state_array[:3]
    velocity = state_array[3:]
    acceleration = gravitational_acceleration(position)

    derivative = np.empty(6, dtype=float)
    derivative[:3] = velocity
    derivative[3:] = acceleration
    return derivative
