"""
Plan a joint-space motion to a TCP pose without moving the robot and visualize the planned
trajectory in Rerun.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python plan_cartesian_pose_in_joint_space.py
    python plan_cartesian_pose_in_joint_space.py --save-rrd plan_cartesian_pose_in_joint_space.rrd
"""

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

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


def configure_series(path: str, name: str, color: list[int]) -> None:
    """Set the name, unit label, color, and width of a scalar plot line."""
    rr.log(
        path,
        rr.SeriesLines(names=[name], colors=[color], widths=[LINE_WIDTH]),
        static=True,
    )


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def visualize_trajectory(
    trajectory: list[tuple[np.ndarray, float]],
    target_joint_positions: np.ndarray,
    app_id: str,
    save_rrd: str | None,
) -> None:
    """Plot the position, velocity, acceleration and error to the target of every joint.

    Args:
        trajectory (list[tuple[np.ndarray, float]]): Planned ``(joint_positions, time)``
            pairs, in degrees and seconds.
        target_joint_positions (np.ndarray): Joint positions the motion ends at, in degrees.
        app_id (str): Rerun application ID of the recording.
        save_rrd (str | None): Path of an .rrd file to save the recording to, or ``None`` to
            open the viewer.
    """
    times = np.array([time for _, time in trajectory])
    positions = np.array([joint_positions for joint_positions, _ in trajectory])
    velocities = finite_difference(positions, times)

    num_joints = positions.shape[1]
    logger.info(f"Planned {len(trajectory)} samples over {times[-1]:.3f} s.")
    logger.info(f"Number of joints: {num_joints}")

    # One horizontal row per joint, each holding its four plots (position, velocity,
    # acceleration, error). Without an explicit blueprint the viewer guesses the layout and
    # drops or merges some plots. Every column shares a fixed Y range across all joints so
    # the plots are directly comparable; the range is taken from the data with a margin.
    signals = {
        "position": positions,
        "velocity": velocities,
        "acceleration": finite_difference(velocities, times),
        "error_to_goal": np.abs(target_joint_positions - positions),
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

    init_rerun(app_id, save_rrd)
    # Force our layout even if the viewer already has a saved blueprint for this recording.
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(*rows)), make_active=True, make_default=True)

    for j in range(num_joints):
        # Position carries two lines: the planned trajectory and the constant target.
        rr.log(
            f"joint_{j}/position",
            rr.SeriesLines(
                names=["planned [deg]", "target [deg]"],
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
            rr.log(
                f"joint_{j}/position",
                rr.Scalars([float(positions[i, j]), float(target_joint_positions[j])]),
            )
            for signal in ("velocity", "acceleration", "error_to_goal"):
                rr.log(f"joint_{j}/{signal}", rr.Scalars(float(signals[signal][i, j])))


def main(save_rrd: str | None = None) -> None:
    """Plan a joint-space motion to a TCP pose and visualize it."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Prepare Target ==========================================
        # Start: the default joint configuration. Target: the active TCP's pose there, 10 cm
        # up in Z.
        q_start = robot.default_joint_configuration
        target_cartesian_pose = robot.forward_kinematics(q_start)
        target_cartesian_pose[2] += 0.1

        # ==================== Run Skill ============================================
        trajectory = robot.plan_cartesian_pose_in_joint_space(
            cartesian_pose=target_cartesian_pose,
            q_start=q_start,
            speed=60,
            acceleration=80,
        )

        # ==================== Visualization (Optional) =============================
        # The last planned configuration is the IK solution for the target pose.
        visualize_trajectory(
            trajectory,
            trajectory[-1][0],
            "telekinesis_synapse_plan_cartesian_pose_in_joint_space",
            save_rrd,
        )
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan Cartesian pose in joint space Synapse example"
    )
    parser.add_argument(
        "--save-rrd",
        type=str,
        default=None,
        help="Save the Rerun recording to this .rrd file instead of opening the viewer",
    )
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
