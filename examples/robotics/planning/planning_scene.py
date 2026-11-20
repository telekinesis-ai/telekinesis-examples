"""
Builds a planning scene around a UR10e, adds a box obstacle and an attached box,
and checks the robot for collisions, visualized in Rerun.

Supports Universal Robots (UR), Epson, and virtual. Runs offline, no hardware needed.

Usage:
    python planning_scene.py
    python planning_scene.py --save-rrd planning_scene.rrd
"""

import argparse

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse import planning_scene
from telekinesis.synapse.robots.manipulators import universal_robots

JOINTS_DEG = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]


def init_rerun(app_id: str, save_rrd: str | None) -> "rr.RecordingStream":
    """Start a Rerun recording, live in the viewer or saved to an .rrd file."""
    rr.init(app_id, spawn=save_rrd is None)
    recording = rr.get_global_data_recording()
    if save_rrd:
        recording.save(save_rrd)
    return recording


def main(save_rrd: str | None) -> None:
    """Add world objects and an attached object, then query collisions."""

    #===================== Create Robot and Scene ================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # The scene copies the robot's kinematic model. The world frame starts at the robot base.
    scene = planning_scene.PlanningScene(kinematic_model=robot.kinematic_model)
    scene.world.add_observer(lambda name, action: logger.info(f"World change: {action} {name}"))

    # ==================== Visualization ========================================
    recording = init_rerun("telekinesis_synapse_planning_scene", save_rrd)
    recording.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Every collision check below is drawn: the robot, the world objects and the attached box,
    # with colliding shapes highlighted. Scrub the "check" timeline to step through them.
    scene.visualize_rerun(recording_stream=recording, retain_collision_checks=True)

    # ==================== Run Skill ============================================
    # Place a 0.2 m box at the TCP pose for this configuration.
    recording.set_time("check", sequence=0)
    world_T_tcp = robot.kinematic_model.forward_kinematics(
        np.deg2rad(JOINTS_DEG), robot.default_tcp
    )
    scene.world.add_box("obstacle", world_T_tcp, [0.2, 0.2, 0.2])
    logger.info(f"World objects: {scene.world.object_names}")
    logger.info(f"Colliding with obstacle: {scene.is_robot_colliding(JOINTS_DEG)}")
    logger.info(f"Colliding geometry: {scene.get_colliding_geometry_names(JOINTS_DEG)}")

    # Move the obstacle 5 m away along x. Absolute pose, in the scene world frame.
    recording.set_time("check", sequence=1)
    far_pose = np.eye(4)
    far_pose[0, 3] = 5.0
    scene.world.set_object_pose("obstacle", far_pose)
    logger.info(f"Colliding after moving it away: {scene.is_robot_colliding(JOINTS_DEG)}")
    scene.world.remove_object("obstacle")

    # Attach a 0.1 m box 0.2 m along the TCP z axis. Touch links are exempt from collisions.
    recording.set_time("check", sequence=2)
    tcp_T_box = np.eye(4)
    tcp_T_box[:3, 3] = [0.0, 0.0, 0.2]
    geometry_names = scene.attach_collision_object(
        name="box",
        parent_frame=robot.default_tcp,
        touch_links=[robot.default_tcp],
        parent_T_object=tcp_T_box,
        bounding_box_size=[0.1, 0.1, 0.1],
    )
    logger.info(f"Attached geometry: {geometry_names}")
    logger.info(f"Attached objects: {scene.attached_collision_object_names}")
    logger.info(f"Colliding with the attached box: {scene.is_robot_colliding(JOINTS_DEG)}")

    # Release the box into the world at its current pose.
    recording.set_time("check", sequence=3)
    scene.detach_collision_object("box", JOINTS_DEG)
    logger.info(f"World objects after detach: {scene.world.object_names}")
    logger.info(f"Colliding after detach: {scene.is_robot_colliding(JOINTS_DEG)}")

    recording.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning scene Synapse example")
    parser.add_argument("--save-rrd", type=str, default=None,
                        help="Save the Rerun recording to this .rrd file instead of opening the viewer")
    args = parser.parse_args()

    main(save_rrd=args.save_rrd)
