import numpy as np
import pytest

from simulation.integrator import propagate_orbit
from simulation.physics import MU_EARTH


def circular_orbit_state() -> np.ndarray:
    radius = 7000.0
    orbital_speed = np.sqrt(MU_EARTH / radius)
    return np.array([radius, 0.0, 0.0, 0.0, orbital_speed, 0.0], dtype=float)


def test_circular_orbit_radius_remains_constant():
    initial_state = circular_orbit_state()
    duration = 2000.0
    time_step = 10.0

    times, states = propagate_orbit(initial_state, duration, time_step)
    radii = np.linalg.norm(states[:, :3], axis=1)

    np.testing.assert_allclose(radii, initial_state[0], rtol=1e-6, atol=1e-3)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(duration)


def test_circular_orbit_period_matches_analytical_period():
    initial_state = circular_orbit_state()
    radius = initial_state[0]
    orbital_speed = initial_state[4]
    analytic_period = 2.0 * np.pi * np.sqrt(radius**3 / MU_EARTH)

    duration = 1.5 * analytic_period
    time_step = 5.0
    times, states = propagate_orbit(initial_state, duration, time_step)

    state_error = states - initial_state
    position_error = np.linalg.norm(state_error[:, :3], axis=1)
    velocity_error = np.linalg.norm(state_error[:, 3:], axis=1)

    return_mask = (
        times > 0.75 * analytic_period
    ) & (position_error < 5e-3 * radius) & (velocity_error < 5e-3 * orbital_speed)

    assert np.any(return_mask), "The orbit did not return to the initial state near one full period."

    returned_index = np.where(return_mask)[0][0]
    returned_time = times[returned_index]
    np.testing.assert_allclose(states[returned_index], initial_state, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(returned_time, analytic_period, rtol=5e-3, atol=5e-3)


def test_specific_orbital_energy_is_conserved():
    initial_state = circular_orbit_state()
    duration = 2000.0
    time_step = 10.0

    times, states = propagate_orbit(initial_state, duration, time_step)
    positions = states[:, :3]
    velocities = states[:, 3:]
    radius = np.linalg.norm(positions, axis=1)
    speeds_squared = np.sum(velocities**2, axis=1)
    energy = 0.5 * speeds_squared - MU_EARTH / radius

    np.testing.assert_allclose(energy, energy[0], rtol=1e-5, atol=1e-4)


def test_specific_angular_momentum_is_conserved():
    initial_state = circular_orbit_state()
    duration = 2000.0
    time_step = 10.0

    _, states = propagate_orbit(initial_state, duration, time_step)
    positions = states[:, :3]
    velocities = states[:, 3:]
    angular_momentum = np.linalg.norm(np.cross(positions, velocities), axis=1)

    np.testing.assert_allclose(angular_momentum, angular_momentum[0], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "initial_state, duration, time_step",
    [
        (np.array([1.0, 2.0, 3.0]), 100.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), -100.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, np.nan, 0.0, 0.0]), 100.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, 0.0, 0.0, np.inf]), 100.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, 0.0, 0.0]), 100.0, 1.0),
        (np.array([7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 100.0, 0.0),
    ],
)
def test_propagate_orbit_rejects_invalid_inputs(initial_state, duration, time_step):
    with pytest.raises((ValueError, RuntimeError)):
        propagate_orbit(initial_state, duration, time_step)


def test_propagate_orbit_returns_regularly_spaced_output():
    initial_state = circular_orbit_state()
    duration = 10.0
    time_step = 0.5

    times, states = propagate_orbit(initial_state, duration, time_step)

    assert times.shape[0] == states.shape[0]
    assert states.shape[1] == 6
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(duration)
    np.testing.assert_allclose(np.diff(times), time_step, rtol=1e-12, atol=1e-12)


def test_propagate_orbit_includes_final_time_for_non_multiple_duration():
    initial_state = circular_orbit_state()
    duration = 10.0
    time_step = 3.0

    times, states = propagate_orbit(initial_state, duration, time_step)

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(duration)
    assert times.shape[0] == states.shape[0]
    assert np.all(np.diff(times) > 0.0)


def test_propagate_orbit_handles_time_step_larger_than_duration():
    initial_state = circular_orbit_state()
    duration = 6.0
    time_step = 12.0

    times, states = propagate_orbit(initial_state, duration, time_step)

    assert times.tolist() == pytest.approx([0.0, duration])
    assert states.shape == (2, 6)
