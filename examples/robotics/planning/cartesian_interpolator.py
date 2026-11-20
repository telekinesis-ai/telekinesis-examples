"""
Plans the same straight-line TCP move with and without a collision check, and compares them,
visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python cartesian_interpolator.py
    python cartesian_interpolator.py --save-rrd cartesian_interpolator.rrd
"""

import argparse

import kinematic_solvers
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from loguru import logger

from telekinesis.synapse import cartesian_interpolator, planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots

START_JOINTS_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def log_tcp_frame(recording: "rr.RecordingStream", path: str, pose: np.ndarray) -> None:
    """Draw an XYZ axis triad at a 4 x 4 TCP pose, in meters."""
    recording.log(path, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]), static=True)
    recording.log(
        f"{path}/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * 0.04,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )


def main(save_rrd: str | None) -> None:
    """Plan a 12 cm move along y, with and without an obstacle check."""

    #===================== Create Robot, Scene and IK Solver =====================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    start_joints_rad = np.deg2rad(START_JOINTS_DEG)

    start_pose = robot.kinematic_model.forward_kinematics(start_joints_rad, robot.default_tcp)
    target_pose = start_pose.copy()
    target_pose[1, 3] += 0.12

    # A small box between the start and the target.
    obstacle_pose = start_pose.copy()
    obstacle_pose[1, 3] += 0.10
    scene = planning_scene.PlanningScene(kinematic_model=robot.kinematic_model)
    scene.world.add_box("obstacle", obstacle_pose, [0.025, 0.025, 0.025])

    # The interpolator needs an IK solver object for the frame it plans.
    solver = kinematic_solvers.TracIkSolver(
        kinematic_model=robot.kinematic_model,
        frame_name=robot.default_tcp,
        timeout=0.05,
    )
    max_step = cartesian_interpolator.MaxEEFStep(translation=0.01, rotation=0.1)
    jump_threshold = cartesian_interpolator.JumpThreshold(4.0)

    # ==================== Run Skill ============================================
    # Without a validity check.
    unchecked = cartesian_interpolator.CartesianInterpolator.compute_cartesian_path(
        kinematic_model=robot.kinematic_model,
        ik_solver=solver,
        start_joints_rad=start_joints_rad,
        frame_name=robot.default_tcp,
        target_pose=target_pose,
        max_step=max_step,
        jump_threshold=jump_threshold,
    )

    # With a validity check. The callback receives radians, but PlanningScene takes degrees.
    def collision_free(q_rad: np.ndarray) -> bool:
        """Accept a configuration when the scene reports no collision."""
        return not scene.is_robot_colliding(np.rad2deg(q_rad))

    checked = cartesian_interpolator.CartesianInterpolator.compute_cartesian_path(
        kinematic_model=robot.kinematic_model,
        ik_solver=solver,
        start_joints_rad=start_joints_rad,
        frame_name=robot.default_tcp,
        target_pose=target_pose,
        max_step=max_step,
        jump_threshold=jump_threshold,
        is_valid=collision_free,
    )

    # The precision overload takes a CartesianPrecision instead of a JumpThreshold.
    precise = cartesian_interpolator.CartesianInterpolator.compute_cartesian_path(
        kinematic_model=robot.kinematic_model,
        ik_solver=solver,
        start_joints_rad=start_joints_rad,
        frame_name=robot.default_tcp,
        target_pose=target_pose,
        max_step=max_step,
        precision=cartesian_interpolator.CartesianPrecision(translational=0.001, rotational=0.01),
    )

    for name, result in (("unchecked", unchecked), ("checked", checked), ("precision", precise)):
        logger.info(f"{name}: fraction={result.fraction:.3f}, "
                    f"states={len(result.joint_trajectory_rad)}")

    # Joint targets for the robot are in degrees.
    last_joints_deg = np.rad2deg(checked.joint_trajectory_rad[-1])
    logger.info(f"Last accepted joints (deg): {np.round(last_joints_deg, 2).tolist()}")

    # ==================== Visualization ========================================
    recording = init_rerun("telekinesis_synapse_cartesian_interpolator", save_rrd)
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/"),
                rrb.TextLogView(origin="/cartesian"),
                column_shares=[3, 1],
            ),
            rrb.TimePanel(timeline="cartesian_step", play_state="following"),
        )
    )
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene.visualize_rerun(recording_stream=recording, retain_collision_checks=True)
    log_tcp_frame(recording, "/cartesian/start_tcp", start_pose)
    log_tcp_frame(recording, "/cartesian/target_tcp", target_pose)

    # Replay each path state by state: the robot at every accepted configuration, with any
    # shape touching the obstacle highlighted. The unchecked path runs through the box.
    sequence = 0
    for name, result in (("unchecked", unchecked), ("checked", checked), ("precision", precise)):
        label = "COMPLETE" if np.isclose(result.fraction, 1.0) else "PARTIAL"
        for index, q_rad in enumerate(result.joint_trajectory_rad):
            recording.set_time("cartesian_step", sequence=sequence)
            sequence += 1
            q_deg = np.rad2deg(q_rad)
            colliding_names = scene.get_colliding_geometry_names(q_deg)
            scene.visualize_configuration_rerun(q_deg, colliding_names)
            recording.log(
                f"/cartesian/{name}/status",
                rr.TextLog(
                    f"{name} {label}; completion={result.fraction:.3f}; "
                    f"state={index + 1}/{len(result.joint_trajectory_rad)}; "
                    f"colliding_geometry_names={colliding_names}"
                ),
            )

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartesian interpolator Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
