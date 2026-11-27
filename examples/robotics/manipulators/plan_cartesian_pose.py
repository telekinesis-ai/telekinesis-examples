"""
Plan a straight-line TCP motion without moving the robot and visualize the planned TCP path in
Rerun.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python plan_cartesian_pose.py
    python plan_cartesian_pose.py --save-rrd plan_cartesian_pose.rrd
"""

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger
from scipy.spatial.transform import Rotation

from telekinesis.synapse.robots.manipulators import universal_robots

# Start and goal TCP poses [x, y, z, rx, ry, rz] in meters and Euler-XYZ degrees.
START_POSE = [0.5, -0.2, 0.4, 180.0, 0.0, 180.0]
GOAL_POSE = [0.4, 0.3, 0.6, 180.0, 30.0, 90.0]

# Joint seed for the IK solve of the start pose, in degrees (UR10E home configuration).
HOME_JOINTS = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]

# TCP limits used to shape the trapezoidal profile.
MAX_LINEAR_VELOCITY = 0.25  # meters per second
MAX_LINEAR_ACCELERATION = 1.0  # meters per second^2

# Width of the plotted lines, in UI points; wider lines are easier to read.
LINE_WIDTH = 3.0

# Plots of the TCP, as (Rerun entity path, title) pairs.
PLOTS = [
    ("cartesian/position/x", "position x [m]"),
    ("cartesian/position/y", "position y [m]"),
    ("cartesian/position/z", "position z [m]"),
    ("cartesian/orientation/x", "orientation x [deg]"),
    ("cartesian/orientation/y", "orientation y [deg]"),
    ("cartesian/orientation/z", "orientation z [deg]"),
    ("cartesian/linear/velocity", "linear velocity [m/s]"),
    ("cartesian/linear/acceleration", "linear acceleration [m/s^2]"),
    ("cartesian/angular/velocity", "angular velocity [deg/s]"),
    ("cartesian/angular/acceleration", "angular acceleration [deg/s^2]"),
    ("cartesian/error/translation", "translation error [m]"),
    ("cartesian/error/orientation", "orientation error [deg]"),
]


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


def orientations_from_poses(poses: np.ndarray) -> Rotation:
    """Convert the Euler-XYZ orientation columns of a pose array to rotations."""
    return Rotation.from_euler("xyz", poses[:, 3:], degrees=True)


def angle_between_deg(a: Rotation, b: Rotation) -> np.ndarray:
    """Angle in degrees of the shortest rotation from each ``a`` to each ``b``."""
    return np.degrees((a.inv() * b).magnitude())


def configure_series(path: str, name: str, color: list[int]) -> None:
    """Set the name, unit label, color, and width of a scalar plot line."""
    rr.log(
        path,
        rr.SeriesLines(names=[name], colors=[color], widths=[LINE_WIDTH]),
        static=True,
    )


def configure_planned_target(path: str, unit: str) -> None:
    """Set up two lines on one plot: the planned trajectory and the constant target value.

    Rerun time-series lines have no dashed style, so the target is drawn as a thinner red
    line rather than a dotted one.
    """
    rr.log(
        path,
        rr.SeriesLines(
            names=[f"planned [{unit}]", f"target [{unit}]"],
            colors=[[31, 119, 180], [214, 39, 40]],
            widths=[LINE_WIDTH, LINE_WIDTH * 0.5],
        ),
        static=True,
    )


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def tcp_signals(tcp_poses: np.ndarray, times: np.ndarray, goal: np.ndarray) -> dict:
    """Values plotted for the TCP at every sample, keyed by the entity paths of their plots.

    Position and orientation plots carry two values per sample, the planned value and the
    constant target; the other plots carry one.

    Args:
        tcp_poses (np.ndarray): TCP poses ``[x, y, z, rx, ry, rz]`` of shape ``(N, 6)``, in
            meters and Euler-XYZ degrees.
        times (np.ndarray): Sample times of shape ``(N,)``, in seconds.
        goal (np.ndarray): Target TCP pose ``[x, y, z, rx, ry, rz]``.

    Returns:
        dict: Arrays of shape ``(N, 2)`` for position and orientation, ``(N,)`` otherwise.
    """
    orientations = orientations_from_poses(tcp_poses)
    # Euler angles are ambiguous modulo 360 degrees, so Euler extraction can flip a
    # component between equivalent branches (e.g. +180 and -180) even when the orientation
    # does not move. Unwrap each component for display and shift the constant target into
    # the same branch as the planned trajectory so the two lines coincide.
    euler_planned = np.unwrap(orientations.as_euler("xyz", degrees=True), period=360.0, axis=0)
    euler_target = goal[3:] + 360.0 * np.round((euler_planned[-1] - goal[3:]) / 360.0)
    # Linear profile: distance from start, and its signed first and second derivatives
    # along the path. Differentiating the scalar progress (not the vector norm) keeps the
    # deceleration phase negative, so the acceleration reads positive / zero / negative.
    linear_speed = finite_difference(
        np.linalg.norm(tcp_poses[:, :3] - tcp_poses[0, :3], axis=1), times
    )
    # Angular profile: cumulative angle travelled, angular speed, and acceleration.
    step_angles = np.zeros(len(times))
    step_angles[1:] = angle_between_deg(orientations[:-1], orientations[1:])
    angular_speed = finite_difference(np.cumsum(step_angles), times)

    signals = {}
    for axis in range(3):
        signals[f"cartesian/position/{'xyz'[axis]}"] = np.column_stack(
            [tcp_poses[:, axis], np.full(len(times), goal[axis])]
        )
        signals[f"cartesian/orientation/{'xyz'[axis]}"] = np.column_stack(
            [euler_planned[:, axis], np.full(len(times), euler_target[axis])]
        )
    signals.update({
        "cartesian/linear/velocity": linear_speed,
        "cartesian/linear/acceleration": finite_difference(linear_speed, times),
        "cartesian/angular/velocity": angular_speed,
        "cartesian/angular/acceleration": finite_difference(angular_speed, times),
        "cartesian/error/translation": np.linalg.norm(goal[:3] - tcp_poses[:, :3], axis=1),
        "cartesian/error/orientation": angle_between_deg(
            orientations, Rotation.from_euler("xyz", goal[3:], degrees=True)
        ),
    })
    return signals


def visualize_tcp_path(
    robot: universal_robots.UniversalRobotsUR10E,
    trajectory: list[tuple[np.ndarray, float]],
    goal: np.ndarray,
    save_rrd: str | None,
) -> None:
    """Plot the TCP path of a planned joint trajectory against the target pose.

    The TCP poses are computed with forward kinematics from the planned joint positions, so
    the plots show where the planned joints really take the TCP.

    Args:
        robot (universal_robots.UniversalRobotsUR10E): Robot the trajectory was planned for.
        trajectory (list[tuple[np.ndarray, float]]): Planned ``(joint_positions, time)``
            pairs, in degrees and seconds.
        goal (np.ndarray): Target TCP pose ``[x, y, z, rx, ry, rz]``.
        save_rrd (str | None): Path of an .rrd file to save the recording to, or ``None`` to
            open the viewer.
    """
    times = np.array([time for _, time in trajectory])
    tcp_poses = np.array([robot.forward_kinematics(q) for q, _ in trajectory])
    orientation_matrices = orientations_from_poses(tcp_poses).as_matrix()
    signals = tcp_signals(tcp_poses, times, goal)
    logger.info(f"Planned {len(trajectory)} samples over {times[-1]:.3f} s.")

    init_rerun("telekinesis_synapse_plan_cartesian_pose", save_rrd)
    # Explicit layout: the 3D path view plus a grid of plots. Without a blueprint the viewer
    # guesses the layout and drops or merges some plots. Force our layout even if the viewer
    # already has a saved blueprint for this recording.
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="TCP path and frame"),
                rrb.Grid(
                    *(rrb.TimeSeriesView(origin=path, name=name) for path, name in PLOTS),
                    grid_columns=3,
                ),
                column_shares=[1, 2],
            )
        ),
        make_active=True,
        make_default=True,
    )

    # 3D view of the straight-line TCP path with start and goal marked.
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

    for axis in "xyz":
        configure_planned_target(f"cartesian/position/{axis}", "m")
        configure_planned_target(f"cartesian/orientation/{axis}", "deg")
    for (path, name), color in zip(
        PLOTS[6:],
        [[255, 127, 14], [44, 160, 44], [255, 127, 14], [44, 160, 44], [214, 39, 40],
         [148, 103, 189]],
    ):
        configure_series(path, name, color)

    for i, t in enumerate(times):
        rr.set_time("time", duration=float(t))
        # Frame travelling from start to goal, showing both position and orientation. The
        # three arrows are the rotated tool X, Y, Z axes (red, green, blue).
        rr.log(
            "world/tcp_frame",
            rr.Arrows3D(
                origins=[tcp_poses[i, :3]] * 3,
                vectors=orientation_matrices[i].T * 0.15,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        for path, values in signals.items():
            rr.log(path, rr.Scalars(np.atleast_1d(values[i]).astype(float)))


def main(save_rrd: str | None = None) -> None:
    """Plan a straight-line TCP motion and visualize it."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Prepare Target ==========================================
        # Start: the joint configuration that puts the TCP at START_POSE, solved from
        # HOME_JOINTS. Target: GOAL_POSE, which also turns the TCP.
        q_start = robot.inverse_kinematics(
            target_poses=[START_POSE], q_inits=[np.asarray(HOME_JOINTS, dtype=float)]
        )[0]
        target_cartesian_pose = np.asarray(GOAL_POSE, dtype=float)

        # ==================== Run Skill ============================================
        trajectory = robot.plan_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            q_start=q_start,
            speed=MAX_LINEAR_VELOCITY,
            acceleration=MAX_LINEAR_ACCELERATION,
        )

        # ==================== Visualization (Optional) =============================
        visualize_tcp_path(robot, trajectory, target_cartesian_pose, save_rrd)
    finally:
        robot.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan Cartesian pose Synapse example")
    parser.add_argument(
        "--save-rrd",
        type=str,
        default=None,
        help="Save the Rerun recording to this .rrd file instead of opening the viewer",
    )
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
