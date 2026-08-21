"""Visualize a LinearCartesianTrajectoryGenerator trajectory in Rerun.

Builds a straight-line (moveL) Cartesian move for a UR10E, resolving every setpoint to
joints with the robot inverse kinematics, then logs the produced trajectory to Rerun so
it can be checked by eye. For each TCP axis it plots the actual position (x, y, z) and
orientation (x, y, z) against the constant target so the convergence to the goal is
visible, together with the linear and angular velocity and acceleration profiles and the
remaining translation and orientation error to the goal. Velocity and acceleration are
computed by finite differences of the sampled poses. A 3D view shows the TCP path with a
coordinate frame travelling from the start pose to the goal pose.

Run with the Rerun viewer:

    python examples/generate_cartesian_trajectory.py
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

from telekinesis.synapse.trajectory_generators.linear_cartesian_trajectory_generator import (
    LinearCartesianTrajectoryGenerator,
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

# Start and goal TCP poses [x, y, z, rx, ry, rz] in meters and Euler-XYZ degrees.
START_POSE = [0.5, -0.2, 0.4, 180.0, 0.0, 180.0]
GOAL_POSE = [0.4, 0.3, 0.6, 180.0, 30.0, 90.0]

# Joint seed for the first IK solve, in degrees (UR10E home configuration).
HOME_JOINTS = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]

# TCP limits used to shape the trapezoidal profile.
MAX_LINEAR_VELOCITY = 0.25  # meters per second
MAX_LINEAR_ACCELERATION = 1.0  # meters per second^2
MAX_ANGULAR_VELOCITY = 45.0  # degrees per second
MAX_ANGULAR_ACCELERATION = 150.0  # degrees per second^2

# Width of the plotted lines, in UI points; wider lines are easier to read.
LINE_WIDTH = 3.0


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


def configure_actual_target(path: str, unit: str) -> None:
    """Set up two lines on one plot: the actual trajectory and the constant target value.

    Rerun time-series lines have no dashed style, so the target is drawn as a thinner red
    line rather than a dotted one.
    """
    rr.log(
        path,
        rr.SeriesLines(
            names=[f"actual [{unit}]", f"target [{unit}]"],
            colors=[[31, 119, 180], [214, 39, 40]],
            widths=[LINE_WIDTH, LINE_WIDTH * 0.5],
        ),
        static=True,
    )


def main():
    #===================== Create Trajectory Generator ===========================
    robot = UniversalRobotsUR10E()
    def ik_resolver(pose, seed): return robot.inverse_kinematics(pose, q_init=seed)

    generator = LinearCartesianTrajectoryGenerator(
        ik_resolver=ik_resolver,
    )

    seed = robot.inverse_kinematics(START_POSE, q_init=np.asarray(HOME_JOINTS, dtype=float))

    # ==================== Run Skill ============================================
    states = generator.generate(
        start_pose=np.asarray(START_POSE, dtype=float),
        goal_pose=np.asarray(GOAL_POSE, dtype=float),
        dt=DT,
        q_start=seed,
        max_linear_velocity=MAX_LINEAR_VELOCITY,
        max_linear_acceleration=MAX_LINEAR_ACCELERATION,
        max_angular_velocity=MAX_ANGULAR_VELOCITY,
        max_angular_acceleration=MAX_ANGULAR_ACCELERATION,
    )

    # ==================== Visualization (Optional) =============================
    times = np.array([state.time for state in states])
    commanded_poses = np.array([state.cartesian_pose for state in states])

    goal = np.asarray(GOAL_POSE, dtype=float)
    commanded_orientations = orientations_from_poses(commanded_poses)
    goal_orientation = Rotation.from_euler("xyz", goal[3:], degrees=True)

    # Linear profile: distance from start, and its signed first and second derivatives
    # along the path. Differentiating the scalar progress (not the vector norm) keeps the
    # deceleration phase negative, so the acceleration reads positive / zero / negative.
    start_position = commanded_poses[0, :3]
    distance_travelled = np.linalg.norm(commanded_poses[:, :3] - start_position, axis=1)
    linear_speed = finite_difference(distance_travelled, times)
    linear_acceleration = finite_difference(linear_speed, times)

    # Angular profile: cumulative angle travelled, angular speed, and acceleration.
    step_angles = np.zeros(len(times))
    step_angles[1:] = angle_between_deg(
        commanded_orientations[:-1], commanded_orientations[1:]
    )
    angle_travelled = np.cumsum(step_angles)
    angular_speed = finite_difference(angle_travelled, times)
    angular_acceleration = finite_difference(angular_speed, times)

    # Remaining error to the goal.
    translation_error = np.linalg.norm(goal[:3] - commanded_poses[:, :3], axis=1)
    angular_error = angle_between_deg(commanded_orientations, goal_orientation)

    # Euler angles are ambiguous modulo 360 degrees, so SLERP followed by Euler extraction
    # can flip a component between equivalent branches (e.g. +180 and -180) even when the
    # orientation does not move. Unwrap each component for display and shift the constant
    # target into the same branch as the actual trajectory so the two lines coincide.
    euler_actual = np.unwrap(
        commanded_orientations.as_euler("xyz", degrees=True),
        period=360.0,
        axis=0,
    )
    euler_target = goal[3:] + 360.0 * np.round((euler_actual[-1] - goal[3:]) / 360.0)

    orientation_matrices = commanded_orientations.as_matrix()

    print(f"Generated {len(states)} samples over {generator.duration:.3f} s.")

    # Explicit layout: the 3D path view plus a grid of plots. Position and orientation are
    # split per axis (x, y, z), each showing the actual trajectory against the constant
    # target so the convergence to the goal is visible. Without a blueprint the viewer
    # guesses the layout and drops or merges some plots.
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="TCP path and frame"),
            rrb.Grid(
                rrb.TimeSeriesView(
                    origin="cartesian/position/x",
                    name="position x [m]"),
                rrb.TimeSeriesView(
                    origin="cartesian/position/y",
                    name="position y [m]"),
                rrb.TimeSeriesView(
                    origin="cartesian/position/z",
                    name="position z [m]"),
                rrb.TimeSeriesView(
                    origin="cartesian/orientation/x",
                    name="orientation x [deg]"),
                rrb.TimeSeriesView(
                    origin="cartesian/orientation/y",
                    name="orientation y [deg]"),
                rrb.TimeSeriesView(
                    origin="cartesian/orientation/z",
                    name="orientation z [deg]"),
                rrb.TimeSeriesView(
                    origin="cartesian/linear/velocity",
                    name="linear velocity [m/s]"),
                rrb.TimeSeriesView(
                    origin="cartesian/linear/acceleration",
                    name="linear acceleration [m/s^2]"),
                rrb.TimeSeriesView(
                    origin="cartesian/angular/velocity",
                    name="angular velocity [deg/s]"),
                rrb.TimeSeriesView(
                    origin="cartesian/angular/acceleration",
                    name="angular acceleration [deg/s^2]"),
                rrb.TimeSeriesView(
                    origin="cartesian/error/translation",
                    name="translation error [m]"),
                rrb.TimeSeriesView(
                    origin="cartesian/error/orientation",
                    name="orientation error [deg]"),
                grid_columns=3,
            ),
            column_shares=[
                1,
                2],
        ))

    rr.init("generate_cartesian_trajectory", spawn=True)
    # Force our layout even if the viewer already has a saved blueprint for this recording.
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    # 3D view of the straight-line TCP path with start and goal marked.
    rr.log(
        "world/tcp_path",
        rr.LineStrips3D([commanded_poses[:, :3]], colors=[[31, 119, 180]], radii=0.004),
        static=True,
    )
    rr.log(
        "world/tcp_path/endpoints",
        rr.Points3D(
            [start_position, goal[:3]],
            colors=[[44, 160, 44], [214, 39, 40]],
            radii=0.012,
            labels=["start", "goal"],
        ),
        static=True,
    )

    for axis in ("x", "y", "z"):
        configure_actual_target(f"cartesian/position/{axis}", "m")
        configure_actual_target(f"cartesian/orientation/{axis}", "deg")
    configure_series("cartesian/linear/velocity", "linear velocity [m/s]", [255, 127, 14])
    configure_series("cartesian/linear/acceleration", "linear acceleration [m/s^2]", [44, 160, 44])
    configure_series("cartesian/angular/velocity", "angular velocity [deg/s]", [255, 127, 14])
    configure_series(
        "cartesian/angular/acceleration", "angular acceleration [deg/s^2]", [44, 160, 44]
    )
    configure_series("cartesian/error/translation", "translation error [m]", [214, 39, 40])
    configure_series("cartesian/error/orientation", "orientation error [deg]", [148, 103, 189])

    axis_length = 0.15
    for i, t in enumerate(times):
        rr.set_time("time", duration=float(t))
        # Frame travelling from start to goal, showing both position and orientation. The
        # three arrows are the rotated tool X, Y, Z axes (red, green, blue).
        origin = commanded_poses[i, :3]
        rr.log(
            "world/tcp_frame",
            rr.Arrows3D(
                origins=[origin, origin, origin],
                vectors=(orientation_matrices[i].T * axis_length),
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        for axis in range(3):
            axis_name = "xyz"[axis]
            rr.log(
                f"cartesian/position/{axis_name}",
                rr.Scalars([float(commanded_poses[i, axis]), float(goal[axis])]),
            )
            rr.log(
                f"cartesian/orientation/{axis_name}",
                rr.Scalars([float(euler_actual[i, axis]), float(euler_target[axis])]),
            )
        rr.log("cartesian/linear/velocity", rr.Scalars(float(linear_speed[i])))
        rr.log("cartesian/linear/acceleration", rr.Scalars(float(linear_acceleration[i])))
        rr.log("cartesian/angular/velocity", rr.Scalars(float(angular_speed[i])))
        rr.log("cartesian/angular/acceleration", rr.Scalars(float(angular_acceleration[i])))
        rr.log("cartesian/error/translation", rr.Scalars(float(translation_error[i])))
        rr.log("cartesian/error/orientation", rr.Scalars(float(angular_error[i])))


if __name__ == "__main__":
    main()
