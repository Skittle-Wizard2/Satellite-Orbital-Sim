import numpy as np
import pytest

from simulation.physics import MU_EARTH, gravitational_acceleration, state_derivative


def test_gravitational_acceleration_at_7000_km():
    position = np.array([7000.0, 0.0, 0.0])
    expected = np.array([-MU_EARTH / 7000.0**2, 0.0, 0.0])

    actual = gravitational_acceleration(position)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "position",
    [
        np.array([7000.0, 0.0, 0.0]),
        np.array([-7000.0, 0.0, 0.0]),
        np.array([0.0, 7000.0, 0.0]),
        np.array([0.0, -7000.0, 0.0]),
        np.array([0.0, 0.0, 7000.0]),
        np.array([0.0, 0.0, -7000.0]),
    ],
)
def test_gravitational_acceleration_direction_along_axes(position):
    actual = gravitational_acceleration(position)
    nonzero = (actual * position)[position != 0.0]

    np.testing.assert_array_less(nonzero, np.zeros_like(nonzero))


@pytest.mark.parametrize("radius", [1000.0, 5000.0, 7000.0, 10000.0, 42164.0])
def test_gravitational_acceleration_magnitude_matches_theory(radius):
    position = np.array([radius, 0.0, 0.0])

    actual = gravitational_acceleration(position)
    expected_magnitude = MU_EARTH / radius**2

    np.testing.assert_allclose(np.linalg.norm(actual), expected_magnitude, rtol=1e-12, atol=1e-12)


def test_state_derivative_correctness():
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

    actual = state_derivative(0.0, state)
    expected_acceleration = gravitational_acceleration(state[:3])
    expected = np.array([0.0, 7.5, 0.0, expected_acceleration[0], expected_acceleration[1], expected_acceleration[2]])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "invalid_input",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
        np.array([np.nan, 0.0, 0.0]),
        np.array([np.inf, 0.0, 0.0]),
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0, 0.0, np.inf]),
    ],
)
def test_gravitational_acceleration_rejects_invalid_vectors(invalid_input):
    with pytest.raises((TypeError, ValueError)):
        gravitational_acceleration(invalid_input)


@pytest.mark.parametrize(
    "invalid_state",
    [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.nan]),
        np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.inf]),
    ],
)
def test_state_derivative_rejects_invalid_states(invalid_state):
    with pytest.raises((TypeError, ValueError)):
        state_derivative(0.0, invalid_state)


def test_velocity_does_not_directly_change_gravity_at_fixed_position():
    position = np.array([7000.0, 0.0, 0.0])
    lower_velocity = np.array([0.0, 0.0, 0.0])
    higher_velocity = np.array([1000.0, -500.0, 250.0])

    state_1 = np.concatenate((position, lower_velocity))
    state_2 = np.concatenate((position, higher_velocity))

    accel_1 = state_derivative(0.0, state_1)[3:]
    accel_2 = state_derivative(0.0, state_2)[3:]

    np.testing.assert_allclose(accel_1, accel_2, rtol=1e-12, atol=1e-12)
