"""
Plan a joint-space motion to a TCP pose without moving the robot and visualize the planned
trajectory in Rerun.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python plan_cartesian_trajectory_in_joint_space.py
    python plan_cartesian_trajectory_in_joint_space.py --save-rrd recording.rrd
"""

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis.synapse.robots.manipulators import universal_robots

# Width of the plotted lines, in UI points; wider lines are easier to read.
LINE_WIDTH = 3.0

# TCP frames drawn along the path: one every this many seconds of motion, with axes this long,
# in meters, and the axis length of the frame that travels along the path.
FRAME_SPACING_S = 0.1
FRAME_AXIS_LENGTH = 0.05
MOVING_FRAME_AXIS_LENGTH = 0.15


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


def orientations_from_poses(poses: np.ndarray) -> Rotation:
    """Convert the Euler-XYZ orientation columns of a pose array to rotations."""
    return Rotation.from_euler("xyz", poses[:, 3:], degrees=True)


def log_tcp_path(tcp_poses: np.ndarray, times: np.ndarray, goal: np.ndarray) -> None:
    """Log the planned TCP path to the 3D view.

    Logs the path line with its start and goal, a TCP frame every ``FRAME_SPACING_S`` seconds
    of motion and at the end, and a larger frame that travels along the path over time. The
    red, green and blue arrows of a frame are the TCP's X, Y and Z axes.

    Args:
        tcp_poses (np.ndarray): Planned TCP poses ``[x, y, z, rx, ry, rz]`` of shape ``(N, 6)``,
            in meters and Euler-XYZ degrees.
        times (np.ndarray): Sample times of shape ``(N,)``, in seconds.
        goal (np.ndarray): Target TCP pose ``[x, y, z, rx, ry, rz]``.
    """
    axis_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    # Row k of each matrix is the TCP's k-th axis in the base frame.
    tcp_axes = orientations_from_poses(tcp_poses).as_matrix().transpose(0, 2, 1)
    rr.log(
        "world/tcp_path",
        rr.LineStrips3D([tcp_poses[:, :3]], colors=[[31, 119, 180]], radii=0.004),
        static=True,
    )
    rr.log(
        "world/tcp_path/endpoints",
        rr.Points3D(
            [tcp_poses[0, :3], goal[:3]],
            colors=[[44, 160, 44], [214, 39, 40]],
            radii=0.012,
            labels=["start", "goal"],
        ),
        static=True,
    )
    # The first sample at or after every FRAME_SPACING_S seconds of motion, and the last one.
    frame_indices = np.unique(
        np.append(
            np.searchsorted(times, np.arange(0.0, times[-1], FRAME_SPACING_S)), len(times) - 1
        )
    )
    rr.log(
        "world/tcp_path/frames",
        rr.Arrows3D(
            origins=np.repeat(tcp_poses[frame_indices, :3], 3, axis=0),
            vectors=tcp_axes[frame_indices].reshape(-1, 3) * FRAME_AXIS_LENGTH,
            colors=axis_colors * len(frame_indices),
        ),
        static=True,
    )
    for i, t in enumerate(times):
        rr.set_time("time", duration=float(t))
        rr.log(
            "world/tcp_frame",
            rr.Arrows3D(
                origins=[tcp_poses[i, :3]] * 3,
                vectors=tcp_axes[i] * MOVING_FRAME_AXIS_LENGTH,
                colors=axis_colors,
            ),
        )


def visualize_trajectory(
    trajectory: list[tuple[np.ndarray, float]],
    target_joint_positions: np.ndarray,
    app_id: str,
    save_rrd: str | None,
) -> None:
    """Plot the position, velocity, acceleration and error to the target of every joint.

    The layout also holds a 3D view of the TCP path on the left, which :func:`log_tcp_path`
    fills.

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
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="TCP path and frames"),
                rrb.Vertical(*rows),
                column_shares=[1, 2],
            )
        ),
        make_active=True,
        make_default=True,
    )

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
        trajectory = robot.plan_cartesian_trajectory_in_joint_space(
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
            "telekinesis_synapse_plan_cartesian_trajectory_in_joint_space",
            save_rrd,
        )
        # The TCP poses the planned joints reach, from forward kinematics.
        log_tcp_path(
            np.array([robot.forward_kinematics(q) for q, _ in trajectory]),
            np.array([time for _, time in trajectory]),
            target_cartesian_pose,
        )
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plan Cartesian trajectory in joint space Synapse example"
    )
    parser.add_argument(
        "--save-rrd",
        type=str,
        default=None,
        help="Save the Rerun recording to this .rrd file instead of opening the viewer",
    )
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
