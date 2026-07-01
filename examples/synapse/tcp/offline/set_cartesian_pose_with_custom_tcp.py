"""
Example: move to 3 world-frame poses using cartesian control, with both a
default and a custom TCP frame.

No hardware required -- runs entirely on the kinematic model with live
visualization in Rerun.

Demonstrates:
  - ``robot.add_tcp()``            -- register a custom TCP offset
  - ``robot.set_cartesian_pose()`` -- move via world-frame target; verify arrival
  - ``robot.active_tcp``           -- switch the active end-effector frame


Run:
    python set_cartesian_pose_with_custom_tcp.py
"""

import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def interpolated_poses(start: list, end: list, n_steps: int) -> list:
    """Return a list of ``n_steps`` linearly interpolated poses from ``start`` to ``end``."""
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    return [
        (start_arr + (step / n_steps) * (end_arr - start_arr)).tolist()
        for step in range(1, n_steps + 1)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # Create robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Visualize the robot in Rerun
    robot.visualize_rerun()

    # Home pose and target cartesian poses
    home_cartesian_pose = [0.40, 0.00, 0.60, 180.0, 0.0, 0.0]
    target_cartesian_poses = [
        [0.40,  0.10,  0.45,  180.0, 0.0,  0.0],
        [0.40, -0.10,  0.45,  180.0, 0.0,  0.0],
        [0.35,  0.00,  0.55,  180.0, 0.0,  0.0],
    ]

    # Build a TF tree for home and target cartesian poses
    tree = tftree.TransformTree("world")

    # Add the home cartesian pose to the TF tree for visualization
    world_T_home = tfutils.pose_to_transformation_matrix(home_cartesian_pose, rot_type="deg")
    tree.add("world", "home", world_T_home, rot_type="mat")

    # Add target cartesian poses to the TF tree for visualization
    for i, target in enumerate(target_cartesian_poses):
        world_T_target = tfutils.pose_to_transformation_matrix(target,
                                                               rot_type="deg")
        tree.add("world",
                 f"target_{i + 1}",
                    world_T_target,
                    rot_type="mat")

    # Visualize the TF tree in Rerun in the same recording stream as the robot
    tree.visualize_rerun(axis_len=0.05, recording_stream=rr.get_global_data_recording())

    # Move to home smoothly to establish a known starting pose
    hz = 60
    n_steps = 40
    for pose in interpolated_poses(robot.get_cartesian_pose(), home_cartesian_pose, n_steps):
        robot.set_cartesian_pose(pose)
        robot.visualize_rerun(recording_stream=rr.get_global_data_recording())
        time.sleep(1.0 / hz)

    # Section 1: cartesian control with the default (tool0) TCP active
    logger.info("\n--- Section 1: cartesian control  (active TCP: '{}') ---", robot.active_tcp)

    # Move the robot to target cartesian poses
    for i, target in enumerate(target_cartesian_poses):
        interpolated_poses_list = interpolated_poses(robot.get_cartesian_pose(), target, n_steps)
        for pose in interpolated_poses_list:
            robot.set_cartesian_pose(pose)
            robot.visualize_rerun(recording_stream=rr.get_global_data_recording())
            time.sleep(1.0 / hz)

    # Return to home with tool0 before switching TCP
    interpolated_poses_list = interpolated_poses(robot.get_cartesian_pose(),
                                                 home_cartesian_pose,
                                                 n_steps)
    for pose in interpolated_poses_list:
        robot.set_cartesian_pose(pose)
        robot.visualize_rerun(recording_stream=rr.get_global_data_recording())
        time.sleep(1.0 / hz)

    # Add gripper
    gripper = onrobot.OnRobotRG6()

    # Attach the gripper and add TCP
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                  set_active=True)
    
    robot.active_tcp = "gripper_tip"
    logger.info("\n--- Section 2: cartesian control  (active TCP: '{}') ---", robot.active_tcp)

    # Move the robot to target cartesian poses with gripper tip as active TCP
    for i, target in enumerate(target_cartesian_poses):
        interpolated_poses_list = interpolated_poses(robot.get_cartesian_pose(),
                                                     target,
                                                     n_steps)
        for pose in interpolated_poses_list:
            robot.set_cartesian_pose(pose)
            robot.visualize_rerun(recording_stream=rr.get_global_data_recording())
            time.sleep(1.0 / hz)

    # Move back home
    interpolated_poses_list = interpolated_poses(robot.get_cartesian_pose(),
                                                 home_cartesian_pose,
                                                 n_steps)
    for pose in interpolated_poses_list:
        robot.set_cartesian_pose(pose)
        robot.visualize_rerun(recording_stream=rr.get_global_data_recording())
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    main()
