"""Visualize a JointTrajectoryGenerator trajectory in Rerun.

Builds a joint-space move for a UR10E with per-joint velocity and acceleration limits,
then logs the produced trajectory to Rerun so it can be checked by eye. For every joint
it plots the position, velocity, acceleration, and remaining error to the goal against a
shared time axis. Velocity and acceleration are computed by finite differences of the
sampled joint positions, so the plots reflect exactly what the generator outputs.

Run with the Rerun viewer:

    python examples/generate_joint_trajectory.py
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from telekinesis.synapse.trajectory_generators.joint_trajectory_generator import (
    JointTrajectoryGenerator,
)
from telekinesis.synapse.robots.manipulators.universal_robots import UniversalRobotsUR10E

def finite_difference(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Numerical time derivative of a sampled signal.

    Differentiates ``values`` with respect to the time vector ``t`` along the sample
    axis using a central difference in the interior and one-sided differences at the
    ends. The time vector may be non-uniform, which matches trajectories whose final
    sample is clamped to the motion duration.

    Args:
        values (np.ndarray): Sampled signal of shape ``(N,)`` or ``(N, D)``, where ``N``
            is the number of samples.
        t (np.ndarray): Sample times of shape ``(N,)``, in seconds.

    Returns:
        np.ndarray: Derivative with the same shape as ``values``. A trajectory with
        fewer than two samples has no defined derivative and yields zeros.
    """
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, t, axis=0)

# Control period the trajectory is sampled at, in seconds.
DT = 0.008

# Start and goal joint configurations in degrees (UR10E has 6 joints).
Q_START = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
Q_GOAL = [90.0, -45.0, -120.0, 45.0, 45.0, -90.0]

# Per-joint limits used to shape the trapezoidal profile.
MAX_JOINT_VELOCITY = [60.0, 45.0, 60.0, 90.0, 90.0, 120.0]  # degrees per second
MAX_JOINT_ACCELERATION = [200.0, 150.0, 200.0, 300.0, 300.0, 400.0]  # degrees per second^2

# Width of the plotted lines, in UI points; wider lines are easier to read.
LINE_WIDTH = 3.0


def shared_axis_range(data: np.ndarray) -> tuple[float, float]:
    """Common Y-axis range covering every value in ``data`` with a small margin."""
    low = float(np.min(data))
    high = float(np.max(data))
    if high - low < 1e-9:
        # Flat signal: open a symmetric window so the line is not pinned to an edge.
        return (low - 1.0, high + 1.0)
    margin = 0.05 * (high - low)
    return (low - margin, high + margin)


def clamp_to_limits(q: np.ndarray, joint_limits: np.ndarray, label: str) -> np.ndarray:
    """Clamp a joint configuration into the robot position limits, warning if it moves."""
    clamped = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])
    if not np.allclose(clamped, q):
        print(f"Warning: {label} was outside the joint limits and has been clamped.")
        print(f"  requested: {q}")
        print(f"  clamped:   {clamped}")
    return clamped


def configure_series(path: str, name: str, color: list[int]) -> None:
    """Set the name, unit label, color, and width of a scalar plot line."""
    rr.log(
        path,
        rr.SeriesLines(names=[name], colors=[color], widths=[LINE_WIDTH]),
        static=True,
    )


def main():
    #===================== Create Trajectory Generator ===========================
    robot = UniversalRobotsUR10E()
    joint_limits = robot.joint_limits

    q_start = clamp_to_limits(np.asarray(Q_START, dtype=float), joint_limits, "Q_START")
    q_goal = clamp_to_limits(np.asarray(Q_GOAL, dtype=float), joint_limits, "Q_GOAL")

    max_joint_velocity = np.asarray(MAX_JOINT_VELOCITY, dtype=float)
    max_joint_acceleration = np.asarray(MAX_JOINT_ACCELERATION, dtype=float)

    generator = JointTrajectoryGenerator()

    # ==================== Run Skill ============================================
    states = generator.generate(
        q_start=q_start,
        q_goal=q_goal,
        dt=DT,
        max_joint_velocity=max_joint_velocity,
        max_joint_acceleration=max_joint_acceleration,
    )

    # ==================== Visualization (Optional) =============================
    times = np.array([state.time for state in states])
    positions = np.array([state.joint_positions for state in states])
    velocities = finite_difference(positions, times)
    accelerations = finite_difference(velocities, times)
    errors = np.abs(q_goal - positions)

    num_joints = positions.shape[1]
    print(f"Generated {len(states)} samples over {generator.duration:.3f} s.")
    print(f'Number of joints: {num_joints}')

    # One horizontal row per joint, each holding its four plots (position, velocity,
    # acceleration, error). Without an explicit blueprint the viewer guesses the layout and
    # drops or merges some plots. Every column shares a fixed Y range across all joints so
    # the plots are directly comparable; the range is taken from the data with a margin.
    signals = {
        "position": positions,
        "velocity": velocities,
        "acceleration": accelerations,
        "error_to_goal": errors,
    }
    axis_ranges = {name: shared_axis_range(data) for name, data in signals.items()}
    rows = [
        rrb.Horizontal(
            *(
                rrb.TimeSeriesView(
                    origin=f"joint_{j}/{signal}",
                    name=f"joint_{j} {signal}",
                    axis_y=rrb.ScalarAxis(range=axis_ranges[signal], zoom_lock=True),
                )
                for signal in signals
            ),
            name=f"joint_{j}",
        )
        for j in range(num_joints)
    ]
    blueprint = rrb.Blueprint(rrb.Vertical(*rows))

    rr.init("generate_joint_trajectory", spawn=True)
    # Force our layout even if the viewer already has a saved blueprint for this recording.
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    for j in range(num_joints):
        # Position carries two lines: the actual trajectory and the constant target (goal).
        rr.log(
            f"joint_{j}/position",
            rr.SeriesLines(
                names=["actual [deg]", "target [deg]"],
                colors=[[31, 119, 180], [214, 39, 40]],
                widths=[LINE_WIDTH, LINE_WIDTH * 0.5],
            ),
            static=True,
        )
        configure_series(f"joint_{j}/velocity", "velocity [deg/s]", [255, 127, 14])
        configure_series(f"joint_{j}/acceleration", "acceleration [deg/s^2]", [44, 160, 44])
        configure_series(f"joint_{j}/error_to_goal", "error to goal [deg]", [214, 39, 40])

    for i, t in enumerate(times):
        rr.set_time("time", duration=float(t))
        for j in range(num_joints):
            rr.log(f"joint_{j}/position", rr.Scalars([float(positions[i, j]), float(q_goal[j])]))
            rr.log(f"joint_{j}/velocity", rr.Scalars(float(velocities[i, j])))
            rr.log(f"joint_{j}/acceleration", rr.Scalars(float(accelerations[i, j])))
            rr.log(f"joint_{j}/error_to_goal", rr.Scalars(float(errors[i, j])))


if __name__ == "__main__":
    main()
