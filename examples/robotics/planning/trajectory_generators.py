"""
Generates a joint-space and a straight-line Cartesian trajectory for a UR10e, and shows the
underlying trapezoidal profile, visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python trajectory_generators.py
    python trajectory_generators.py --save-rrd trajectory_generators.rrd

Note:
    For fuller plotted versions with velocity, acceleration and error curves, see
    manipulators/advanced/generate_joint_trajectory.py and
    manipulators/advanced/generate_cartesian_trajectory.py.
"""

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger

from telekinesis.synapse import planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.trajectory_generators.joint_trajectory_generator import (
    JointTrajectoryGenerator,
)
from telekinesis.synapse.trajectory_generators.linear_cartesian_trajectory_generator import (
    LinearCartesianTrajectoryGenerator,
)
from telekinesis.synapse.trajectory_generators.trapezoidal_profile import TrapezoidalProfile

DT = 0.008  # control period in seconds

# Joint move, in degrees.
Q_START = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]
Q_GOAL = [90.0, -45.0, -120.0, 45.0, 45.0, -90.0]
MAX_JOINT_VELOCITY = [60.0, 45.0, 60.0, 90.0, 90.0, 120.0]  # deg/s, one per joint
MAX_JOINT_ACCELERATION = [200.0, 150.0, 200.0, 300.0, 300.0, 400.0]  # deg/s^2, one per joint

# Cartesian move: [x, y, z, rx, ry, rz] in meters and Euler XYZ degrees.
START_POSE = [0.5, -0.2, 0.4, 180.0, 0.0, 180.0]
GOAL_POSE = [0.4, 0.3, 0.6, 180.0, 30.0, 90.0]
HOME_JOINTS = [0.0, -90.0, -90.0, 0.0, 90.0, 0.0]  # IK seed for the first setpoint

# Redraw the robot every this many samples, to keep the recording small.
DRAW_EVERY = 10
# Width of the plotted lines, in UI points.
LINE_WIDTH = 3.0
JOINT_COLORS = [[31, 119, 180], [255, 127, 14], [44, 160, 44],
                [214, 39, 40], [148, 103, 189], [140, 86, 75]]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_tcp_frame(path: str, pose: np.ndarray) -> None:
    """Draw an XYZ axis triad at a 4 x 4 TCP pose, in meters, at the current time."""
    rr.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
    rr.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.05,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )


def main(save_rrd: str | None) -> None:
    """Generate both trajectories and log their duration and final state."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    model = robot.kinematic_model
    frame = robot.default_tcp

    # ==================== Joint Trajectory =====================================
    joint_generator = JointTrajectoryGenerator()
    joint_states = joint_generator.generate(
        q_start=np.asarray(Q_START, dtype=float),
        q_goal=np.asarray(Q_GOAL, dtype=float),
        dt=DT,
        max_joint_velocity=np.asarray(MAX_JOINT_VELOCITY, dtype=float),
        max_joint_acceleration=np.asarray(MAX_JOINT_ACCELERATION, dtype=float),
    )
    logger.info(f"Joint trajectory: {len(joint_states)} samples over {joint_generator.duration:.3f} s")
    logger.info(f"Last state: t={joint_states[-1].time:.3f} s, final={joint_states[-1].is_final}, "
                f"joints={np.round(joint_states[-1].joint_positions, 2).tolist()}")

    # ==================== Linear Cartesian Trajectory ==========================
    # The generator calls this for every setpoint. Return joints in degrees, or None
    # when there is no solution.
    def ik_resolver(pose, seed):
        return robot.inverse_kinematics([pose], q_inits=[seed])[0]

    cartesian_generator = LinearCartesianTrajectoryGenerator(ik_resolver=ik_resolver)
    seed = ik_resolver(START_POSE, np.asarray(HOME_JOINTS, dtype=float))
    cartesian_states = cartesian_generator.generate(
        start_pose=np.asarray(START_POSE, dtype=float),
        goal_pose=np.asarray(GOAL_POSE, dtype=float),
        dt=DT,
        q_start=seed,
        max_linear_velocity=0.25,
        max_linear_acceleration=1.0,
        max_angular_velocity=45.0,
        max_angular_acceleration=150.0,
    )
    logger.info(f"Cartesian trajectory: {len(cartesian_states)} samples "
                f"over {cartesian_generator.duration:.3f} s")
    logger.info(f"Last pose: {np.round(cartesian_states[-1].cartesian_pose, 3).tolist()}")

    # ==================== Trapezoidal Profile ==================================
    profile = TrapezoidalProfile(v_max=1.0, a=4.0)
    logger.info(f"Profile: duration={profile.duration:.3f} s, peak velocity={profile.peak_velocity:.3f}, "
                f"acceleration={profile.acceleration:.3f}")
    logger.info(f"Position at mid-time: {profile.get_position(profile.duration / 2):.3f}")

    # ==================== Visualization ========================================
    # One timeline, in seconds: the joint trajectory first, then the Cartesian trajectory,
    # then the profile, each starting one second after the previous one ends.
    cartesian_start_s = joint_generator.duration + 1.0
    profile_start_s = cartesian_start_s + cartesian_generator.duration + 1.0

    recording = init_rerun("telekinesis_synapse_trajectory_generators", save_rrd)
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/", name="Robot and TCP paths"),
                rrb.Grid(
                    rrb.TimeSeriesView(origin="/joint_trajectory/positions", name="joint positions [deg]"),
                    rrb.TimeSeriesView(origin="/cartesian_trajectory/position", name="TCP position [m]"),
                    rrb.TimeSeriesView(origin="/profile/position", name="trapezoidal profile"),
                    grid_columns=1,
                ),
                column_shares=[2, 1],
            ),
            rrb.TimePanel(timeline="time", play_state="following"),
        )
    )
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # A planning scene draws the robot at the sampled configurations.
    scene = planning_scene.PlanningScene(kinematic_model=model)
    scene.visualize_rerun(recording_stream=recording)

    rr.log(
        "/joint_trajectory/positions",
        rr.SeriesLines(names=[f"joint {i + 1}" for i in range(6)],
                       colors=JOINT_COLORS, widths=[LINE_WIDTH] * 6),
        static=True,
    )
    rr.log(
        "/cartesian_trajectory/position",
        rr.SeriesLines(names=["x", "y", "z"], colors=JOINT_COLORS[:3], widths=[LINE_WIDTH] * 3),
        static=True,
    )
    rr.log(
        "/profile/position",
        rr.SeriesLines(names=["normalized position"], colors=[[31, 119, 180]], widths=[LINE_WIDTH]),
        static=True,
    )

    # The joint trajectory: the TCP path it traces, the joint positions, and the robot moving.
    joint_tcp_path = np.array(
        [model.forward_kinematics(np.deg2rad(state.joint_positions), frame)[:3, 3] for state in joint_states]
    )
    rr.log("/joint_trajectory/tcp_path",
           rr.LineStrips3D([joint_tcp_path], colors=[[31, 119, 180]], radii=0.004), static=True)
    for index, state in enumerate(joint_states):
        rr.set_time("time", duration=float(state.time))
        rr.log("/joint_trajectory/positions", rr.Scalars([float(v) for v in state.joint_positions]))
        if index % DRAW_EVERY == 0 or state.is_final:
            scene.visualize_configuration_rerun(state.joint_positions, [])
            log_tcp_frame("/joint_trajectory/tcp_frame",
                          model.forward_kinematics(np.deg2rad(state.joint_positions), frame))

    # The Cartesian trajectory: the straight TCP path, the commanded position, and the robot moving.
    cartesian_tcp_path = np.array([state.cartesian_pose[:3] for state in cartesian_states])
    rr.log("/cartesian_trajectory/tcp_path",
           rr.LineStrips3D([cartesian_tcp_path], colors=[[255, 127, 14]], radii=0.004), static=True)
    for index, state in enumerate(cartesian_states):
        rr.set_time("time", duration=float(cartesian_start_s + state.time))
        rr.log("/cartesian_trajectory/position", rr.Scalars([float(v) for v in state.cartesian_pose[:3]]))
        if index % DRAW_EVERY == 0 or state.is_final:
            scene.visualize_configuration_rerun(state.joint_positions, [])
            log_tcp_frame("/cartesian_trajectory/tcp_frame",
                          model.forward_kinematics(np.deg2rad(state.joint_positions), frame))

    # The trapezoidal profile on its own: a normalized position from 0 to 1.
    for t in np.linspace(0.0, profile.duration, 100):
        rr.set_time("time", duration=float(profile_start_s + t))
        rr.log("/profile/position", rr.Scalars(float(profile.get_position(float(t)))))

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory generators Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
