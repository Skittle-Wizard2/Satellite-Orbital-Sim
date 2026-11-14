from __future__ import annotations

import numpy as np

from simulation.physics import MU_EARTH


def orbital_elements_to_state(
    a: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    omega_deg: float,
    nu_deg: float,
) -> np.ndarray:
    """Convert classical orbital elements to Earth-centered Cartesian state vectors.

    The input orbit is restricted to elliptic two-body motion about Earth. The
    returned state vector is ordered as [x, y, z, vx, vy, vz], with position in km
    and velocity in km/s.

    Args:
        a: Semi-major axis in km. Must be greater than zero.
        e: Eccentricity. Must satisfy 0 <= e < 1.
        i_deg: Inclination in degrees. Must lie in [0, 180].
        raan_deg: Right ascension of ascending node in degrees.
        omega_deg: Argument of periapsis in degrees.
        nu_deg: True anomaly in degrees.

    Returns:
        A six-element NumPy array containing the Cartesian state vector.

    Raises:
        ValueError: If any orbital element is invalid for an elliptic orbit.
    """
    values = np.asarray([a, e, i_deg, raan_deg, omega_deg, nu_deg], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("All orbital elements must be finite numbers.")

    if a <= 0.0:
        raise ValueError("Semi-major axis a must be greater than zero.")

    if not 0.0 <= e < 1.0:
        raise ValueError("Eccentricity e must satisfy 0 <= e < 1 for elliptic orbits.")

    if not 0.0 <= i_deg <= 180.0:
        raise ValueError("Inclination i must be between 0 and 180 degrees inclusive.")

    i = np.deg2rad(i_deg)
    raan = np.deg2rad(raan_deg)
    omega = np.deg2rad(omega_deg)
    nu = np.deg2rad(nu_deg)

    p = a * (1.0 - e**2)
    r_pqw = np.array(
        [
            p * np.cos(nu) / (1.0 + e * np.cos(nu)),
            p * np.sin(nu) / (1.0 + e * np.cos(nu)),
            0.0,
        ],
        dtype=float,
    )

    v_pqw = np.array(
        [
            -np.sqrt(MU_EARTH / p) * np.sin(nu),
            np.sqrt(MU_EARTH / p) * (e + np.cos(nu)),
            0.0,
        ],
        dtype=float,
    )

    rotation_3 = np.array(
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    rotation_1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i), np.cos(i)],
        ],
        dtype=float,
    )

    rotation_3_omega = np.array(
        [
            [np.cos(omega), -np.sin(omega), 0.0],
            [np.sin(omega), np.cos(omega), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    rotation_matrix = rotation_3 @ rotation_1 @ rotation_3_omega
    position = rotation_matrix @ r_pqw
    velocity = rotation_matrix @ v_pqw

    return np.concatenate((position, velocity))
