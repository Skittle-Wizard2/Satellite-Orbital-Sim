import numpy as np
import pytest

from simulation.integrator import propagate_orbit
from simulation.orbital_elements import orbital_elements_to_state
from simulation.physics import MU_EARTH


def test_circular_equatorial_orbit():
    a = 7000.0
    e = 0.0
    state = orbital_elements_to_state(a, e, 0.0, 0.0, 0.0, 0.0)

    expected_position = np.array([a, 0.0, 0.0], dtype=float)
    expected_velocity = np.array([0.0, np.sqrt(MU_EARTH / a), 0.0], dtype=float)

    np.testing.assert_allclose(state[:3], expected_position, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state[3:], expected_velocity, rtol=1e-12, atol=1e-12)


def test_inclined_circular_orbit_orientation():
    a = 7000.0
    e = 0.0
    i_deg = 45.0
    raan_deg = 30.0
    omega_deg = 60.0
    nu_deg = 0.0

    state = orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)

    i = np.deg2rad(i_deg)
    raan = np.deg2rad(raan_deg)
    omega = np.deg2rad(omega_deg)

    rotation = np.array(
        [
            [np.cos(raan) * np.cos(omega) - np.sin(raan) * np.sin(omega) * np.cos(i),
             -np.cos(raan) * np.sin(omega) - np.sin(raan) * np.cos(omega) * np.cos(i),
             np.sin(raan) * np.sin(i)],
            [np.sin(raan) * np.cos(omega) + np.cos(raan) * np.sin(omega) * np.cos(i),
             -np.sin(raan) * np.sin(omega) + np.cos(raan) * np.cos(omega) * np.cos(i),
             -np.cos(raan) * np.sin(i)],
            [np.sin(omega) * np.sin(i),
             np.cos(omega) * np.sin(i),
             np.cos(i)],
        ],
        dtype=float,
    )

    expected_position = a * rotation[:, 0]
    expected_velocity = np.sqrt(MU_EARTH / a) * rotation[:, 1]

    np.testing.assert_allclose(state[:3], expected_position, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state[3:], expected_velocity, rtol=1e-12, atol=1e-12)


def test_elliptical_orbit_position_and_velocity_magnitudes():
    a = 7000.0
    e = 0.2
    i_deg = 30.0
    raan_deg = 40.0
    omega_deg = 20.0
    nu_deg = 60.0

    state = orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)
    position = state[:3]
    velocity = state[3:]

    expected_radius = a * (1.0 - e**2) / (1.0 + e * np.cos(np.deg2rad(nu_deg)))
    expected_speed_squared = MU_EARTH * (2.0 / expected_radius - 1.0 / a)

    np.testing.assert_allclose(np.linalg.norm(position), expected_radius, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(velocity) ** 2, expected_speed_squared, rtol=1e-12, atol=1e-12)


def test_generate_state_matches_propagate_orbit():
    a = 7000.0
    e = 0.1
    i_deg = 15.0
    raan_deg = 35.0
    omega_deg = 55.0
    nu_deg = 90.0

    initial_state = orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)
    times, states = propagate_orbit(initial_state, 5.0, 0.5)

    np.testing.assert_allclose(states[0], initial_state, rtol=1e-12, atol=1e-12)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(5.0)
    assert states.shape[1] == 6


@pytest.mark.parametrize(
    "a, e, i_deg, raan_deg, omega_deg, nu_deg",
    [
        (0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        (7000.0, -0.1, 0.0, 0.0, 0.0, 0.0),
        (7000.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (7000.0, 0.5, 181.0, 0.0, 0.0, 0.0),
        (7000.0, 0.5, 0.0, np.nan, 0.0, 0.0),
        (7000.0, 0.5, 0.0, 0.0, np.inf, 0.0),
    ],
)
def test_orbital_elements_rejects_invalid_inputs(a, e, i_deg, raan_deg, omega_deg, nu_deg):
    with pytest.raises(ValueError):
        orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)


def test_orbital_elements_handles_eccentricity_and_orientation_consistently():
    a = 8000.0
    e = 0.3
    i_deg = 50.0
    raan_deg = 120.0
    omega_deg = 25.0
    nu_deg = 75.0

    state = orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)
    position = state[:3]
    velocity = state[3:]

    expected_radius = a * (1.0 - e**2) / (1.0 + e * np.cos(np.deg2rad(nu_deg)))
    expected_speed_squared = MU_EARTH * (2.0 / expected_radius - 1.0 / a)

    np.testing.assert_allclose(np.linalg.norm(position), expected_radius, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(velocity) ** 2, expected_speed_squared, rtol=1e-12, atol=1e-12)
    assert np.isfinite(position).all()
    assert np.isfinite(velocity).all()


def test_orbital_elements_are_invariant_under_angular_wrapping():
    a = 7000.0
    e = 0.2
    i_deg = 35.0
    raan_deg = 40.0
    omega_deg = 50.0
    nu_deg = 60.0

    base_state = orbital_elements_to_state(a, e, i_deg, raan_deg, omega_deg, nu_deg)
    wrapped_state = orbital_elements_to_state(
        a,
        e,
        i_deg,
        raan_deg + 360.0,
        omega_deg + 360.0,
        nu_deg + 360.0,
    )

    np.testing.assert_allclose(wrapped_state, base_state, rtol=1e-12, atol=1e-12)


def test_orbital_elements_eccentricity_boundary_cases():
    a = 7000.0
    e_circular = 0.0
    e_high = 0.999

    circular_state = orbital_elements_to_state(a, e_circular, 25.0, 10.0, 15.0, 30.0)
    high_e_state = orbital_elements_to_state(a, e_high, 25.0, 10.0, 15.0, 30.0)

    radius_circular = np.linalg.norm(circular_state[:3])
    radius_high = np.linalg.norm(high_e_state[:3])
    speed_circular = np.linalg.norm(circular_state[3:])
    speed_high = np.linalg.norm(high_e_state[3:])

    np.testing.assert_allclose(radius_circular, a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(speed_circular**2, MU_EARTH / a, rtol=1e-12, atol=1e-12)

    expected_radius_high = a * (1.0 - e_high**2) / (1.0 + e_high * np.cos(np.deg2rad(30.0)))
    expected_speed_high_squared = MU_EARTH * (2.0 / expected_radius_high - 1.0 / a)

    np.testing.assert_allclose(radius_high, expected_radius_high, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(speed_high**2, expected_speed_high_squared, rtol=1e-12, atol=1e-12)
    assert np.isfinite(circular_state).all()
    assert np.isfinite(high_e_state).all()
