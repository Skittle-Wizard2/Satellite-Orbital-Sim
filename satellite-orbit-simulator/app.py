from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from simulation.integrator import propagate_orbit
from simulation.orbital_elements import orbital_elements_to_state
from simulation.physics import MU_EARTH

EARTH_RADIUS_KM = 6371.0
PLAYBACK_SPEEDS = {
    "0.5x": 2000,
    "1x": 1000,
    "2x": 500,
    "4x": 250,
}


def _validate_orbit_inputs(
    a: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    omega_deg: float,
    nu_deg: float,
    duration: float,
    time_step: float,
) -> list[str]:
    """Validate Earth-centered elliptic-orbit inputs before propagation."""
    errors: list[str] = []

    for name, value in {
        "Semi-major axis": a,
        "Eccentricity": e,
        "Inclination": i_deg,
        "RAAN": raan_deg,
        "Argument of periapsis": omega_deg,
        "True anomaly": nu_deg,
        "Duration": duration,
        "Time step": time_step,
    }.items():
        if not np.isfinite(value):
            errors.append(f"{name} must be finite.")

    if a <= 0.0:
        errors.append("Semi-major axis must be greater than zero.")

    if not 0.0 <= e < 1.0:
        errors.append("Eccentricity must satisfy 0 <= e < 1.")

    if not 0.0 <= i_deg <= 180.0:
        errors.append("Inclination must be between 0 and 180 degrees.")

    if duration <= 0.0:
        errors.append("Simulation duration must be positive.")

    if time_step <= 0.0:
        errors.append("Output time step must be positive.")

    return errors


def _orbital_period(a_km: float) -> float:
    """Return the orbital period in seconds using the Kepler relation."""
    return 2.0 * np.pi * np.sqrt(a_km**3 / MU_EARTH)


def _earth_surface_mesh(radius: float = EARTH_RADIUS_KM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 40j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    return x, y, z


def _build_animation_figure(
    trajectory: np.ndarray,
    times: np.ndarray,
    initial_state: np.ndarray,
    speed_ms: int,
    selected_frame: int = 0,
    earth_radius: float = EARTH_RADIUS_KM,
) -> go.Figure:
    positions = trajectory[:, :3]
    max_radius = max(earth_radius * 1.5, float(np.max(np.linalg.norm(positions, axis=1))) * 1.25)

    x_surface, y_surface, z_surface = _earth_surface_mesh(earth_radius)
    x_values = positions[:, 0]
    y_values = positions[:, 1]
    z_values = positions[:, 2]

    earth_trace = go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        opacity=0.55,
        colorscale=[[0, "#3d7dd8"], [1, "#163d78"]],
        showscale=False,
        hoverinfo="skip",
    )
    orbit_trace = go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        line=dict(color="#ff6b6b", width=4),
        name="Orbit trajectory",
    )
    initial_marker = go.Scatter3d(
        x=[initial_state[0]],
        y=[initial_state[1]],
        z=[initial_state[2]],
        mode="markers",
        marker=dict(size=8, color="#00ff88"),
        name="Initial",
    )
    traveled_trace = go.Scatter3d(
        x=x_values[:1],
        y=y_values[:1],
        z=z_values[:1],
        mode="lines",
        line=dict(color="#ffb703", width=4),
        name="Travelled path",
    )
    satellite_marker = go.Scatter3d(
        x=[x_values[0]],
        y=[y_values[0]],
        z=[z_values[0]],
        mode="markers",
        marker=dict(size=10, color="#ffd166"),
        name="Satellite",
    )

    frames = []
    for index in range(len(times)):
        trail_x = x_values[: index + 1]
        trail_y = y_values[: index + 1]
        trail_z = z_values[: index + 1]
        frame_data = [
            go.Scatter3d(
                x=trail_x,
                y=trail_y,
                z=trail_z,
                mode="lines",
                line=dict(color="#ffb703", width=4),
            ),
            go.Scatter3d(
                x=[x_values[index]],
                y=[y_values[index]],
                z=[z_values[index]],
                mode="markers",
                marker=dict(size=10, color="#ffd166"),
            ),
        ]
        frames.append(go.Frame(data=frame_data, traces=[3, 4], name=str(index)))

    fig = go.Figure(
        data=[earth_trace, orbit_trace, initial_marker, traveled_trace, satellite_marker],
        frames=frames,
    )

    fig.update_layout(
        title="Earth-Centered Orbit Animation",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="cube",
            xaxis=dict(range=[-max_radius, max_radius]),
            yaxis=dict(range=[-max_radius, max_radius]),
            zaxis=dict(range=[-max_radius, max_radius]),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": speed_ms, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": max(0, min(selected_frame, len(times) - 1)),
                "pad": {"t": 30},
                "currentvalue": {"prefix": "Elapsed time: ", "suffix": " s"},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(index)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": speed_ms, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{times[index]:.0f}",
                    }
                    for index in range(len(times))
                ],
            }
        ],
    )
    return fig


st.set_page_config(page_title="Satellite Orbit Simulator", layout="wide")
st.title("Satellite Orbit Simulator")
st.caption("Interactive Earth-centered orbit visualization using the existing orbital mechanics modules.")

if "orbit_cache" not in st.session_state:
    st.session_state.orbit_cache = None
if "timeline_index" not in st.session_state:
    st.session_state.timeline_index = 0
if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = "1x"

with st.form("orbit_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        a_km = st.number_input("Semi-major axis (km)", value=7000.0, min_value=1000.0, step=100.0)
        e_value = st.number_input("Eccentricity", value=0.001, min_value=0.0, max_value=0.999999, step=0.001)
        i_deg = st.number_input("Inclination (deg)", value=51.6, min_value=0.0, max_value=180.0, step=0.1)
    with col2:
        raan_deg = st.number_input("RAAN (deg)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
        omega_deg = st.number_input("Argument of periapsis (deg)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
        nu_deg = st.number_input("True anomaly (deg)", value=0.0, min_value=0.0, max_value=360.0, step=1.0)
    with col3:
        duration = st.number_input("Simulation duration (s)", value=5400.0, min_value=1.0, step=10.0)
        time_step = st.number_input("Output time step (s)", value=30.0, min_value=0.1, step=1.0)

    submitted = st.form_submit_button("Generate Orbit")

if submitted:
    errors = _validate_orbit_inputs(a_km, e_value, i_deg, raan_deg, omega_deg, nu_deg, duration, time_step)
    if errors:
        st.error("Input validation failed:")
        for error in errors:
            st.write(f"- {error}")
        st.stop()

    try:
        initial_state = orbital_elements_to_state(a_km, e_value, i_deg, raan_deg, omega_deg, nu_deg)
        times, states = propagate_orbit(initial_state, duration, time_step)
    except Exception as exc:  # pragma: no cover - UI error handling path
        st.error(f"Simulation failed: {exc}")
        st.stop()

    st.session_state.orbit_cache = {
        "a_km": float(a_km),
        "initial_state": initial_state.copy(),
        "times": times.copy(),
        "states": states.copy(),
        "duration": float(duration),
        "time_step": float(time_step),
    }
    st.session_state.timeline_index = 0
    st.session_state.playback_speed = "1x"

orbit_cache = st.session_state.orbit_cache
if orbit_cache is None:
    st.info("Configure the orbit and click Generate Orbit to view the propagated trajectory.")
    st.stop()

initial_state = orbit_cache["initial_state"]
times = orbit_cache["times"]
states = orbit_cache["states"]
final_state = states[-1]
initial_position = initial_state[:3]
final_position = final_state[:3]
final_velocity = final_state[3:]

orbital_period = _orbital_period(orbit_cache["a_km"])
final_distance = float(np.linalg.norm(final_position))
final_altitude = final_distance - EARTH_RADIUS_KM

playback_speed_label = st.selectbox(
    "Playback speed",
    options=list(PLAYBACK_SPEEDS),
    index=list(PLAYBACK_SPEEDS).index(st.session_state.playback_speed),
    key="playback_speed",
)
if st.button("Reset animation"):
    st.session_state.timeline_index = 0

timeline_index = st.slider(
    "Timeline",
    min_value=0,
    max_value=max(len(times) - 1, 0),
    value=st.session_state.timeline_index,
    key="timeline_index",
    help="Select the propagated sample used for the snapshot metrics.",
)

current_time = float(times[timeline_index])
current_state = states[timeline_index]
current_position = current_state[:3]
current_velocity = current_state[3:]
current_distance = float(np.linalg.norm(current_position))
current_altitude = current_distance - EARTH_RADIUS_KM

fig = _build_animation_figure(
    states[:, :3],
    times,
    initial_state,
    PLAYBACK_SPEEDS[playback_speed_label],
    selected_frame=timeline_index,
)
fig.layout.sliders[0].active = timeline_index
st.plotly_chart(fig, use_container_width=True)

st.subheader("Orbit state summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Elapsed time", f"{current_time:.1f} s")
    st.metric("Current position", f"({current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}) km")
    st.metric("Initial position", f"({initial_position[0]:.2f}, {initial_position[1]:.2f}, {initial_position[2]:.2f}) km")
with col2:
    st.metric("Current velocity", f"({current_velocity[0]:.4f}, {current_velocity[1]:.4f}, {current_velocity[2]:.4f}) km/s")
    st.metric("Final velocity", f"({final_velocity[0]:.4f}, {final_velocity[1]:.4f}, {final_velocity[2]:.4f}) km/s")
    st.metric("Initial orbital period", f"{orbital_period:.2f} s")
with col3:
    st.metric("Distance from center", f"{current_distance:.2f} km")
    st.metric("Altitude above surface", f"{current_altitude:.2f} km")
    st.metric("Final position", f"({final_position[0]:.2f}, {final_position[1]:.2f}, {final_position[2]:.2f}) km")

st.write("Trajectory samples:", len(times))
st.write("Time range:", f"{times[0]:.2f} s to {times[-1]:.2f} s")
