"""
Example: visualize a UR10e robot with a Robotiq gripper in Rerun.

Supported grippers:
  - 2f85     Robotiq 2F-85  (default)
  - 2f140    Robotiq 2F-140
  - hande    Robotiq Hand-E

Demonstrates:
  - ``robot.attach_tool(gripper)``     -- attach gripper once; visualization is automatic
  - ``robot.visualize_rerun()``        -- renders robot + gripper together every step
  - TF tree with home and target poses visualized as coordinate frames
  - Linear interpolation over Cartesian targets
  
Supported for all robots offline, and Universal Robots in real.

Run:
    python examples/synapse/attach_tool/attach_and_move_with_robotiq.py                # defaults to 2F-85
    python examples/synapse/attach_tool/attach_and_move_with_robotiq.py --gripper 2f140
    python examples/synapse/attach_tool/attach_and_move_with_robotiq.py --gripper hande
"""

import argparse
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.tf import tftree, tfutils
from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import robotiq

_GRIPPERS = {
    "2f85":  (robotiq.Robotiq2F85,   0.100),
    "2f140": (robotiq.Robotiq2F140,  0.130),
    "hande": (robotiq.RobotiqHandE,  0.150), 
}


def interpolated_poses(start: list, end: list, n_steps: int) -> list:
    """Return ``n_steps`` linearly interpolated poses from ``start`` to ``end``."""

    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    return [
        (start_arr + (step / n_steps) * (end_arr - start_arr)).tolist()
        for step in range(1, n_steps + 1)
    ]


def main():
    """
    Visualize a UR10e robot with a Robotiq gripper in Rerun,
    and move the robot through a series of Cartesian poses.
    """

    parser = argparse.ArgumentParser(description="UR10e + Robotiq gripper Rerun visualization")
    parser.add_argument(
        "--gripper",
        choices=list(_GRIPPERS),
        default="2f85",
        help="Gripper model to visualize (default: 2f85)",
    )
    args = parser.parse_args()
    gripper_cls, tcp_z = _GRIPPERS[args.gripper]


    # Create the robot and gripper instances
    robot = universal_robots.UniversalRobotsUR10E()
    gripper = gripper_cls()

    # Attach the gripper to the robot and set the active TCP frame
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, tcp_z, 0.0, 0.0, 0.0],
                  set_active=True)


    # Define home and target cartesian poses
    home_cartesian_pose = [0.40, 0.00, 0.60, 180.0, 0.0, 0.0]
    target_cartesian_poses = [
        [0.40,  0.10,  0.45, 180.0, 0.0, 0.0],
        [0.40, -0.10,  0.45, 180.0, 0.0, 0.0],
        [0.35,  0.00,  0.55, 180.0, 0.0, 0.0],
    ]

    # Visualize the TF tree with home and target poses in Rerun,
    # and visualize the robot + gripper together
    rr.init(f"telekinesis_synapse_ur10e_robotiq_{args.gripper}", spawn=True)
    recording_stream = rr.get_global_data_recording()

    tree = tftree.TransformTree("world")
    tree.add("world", "home",
             tfutils.pose_to_transformation_matrix(home_cartesian_pose, rot_type="deg"),
             rot_type="mat")
    for i, target in enumerate(target_cartesian_poses):
        tree.add("world", f"target_{i + 1}",
                 tfutils.pose_to_transformation_matrix(target, rot_type="deg"),
                 rot_type="mat")
    tree.visualize_rerun(axis_len=0.05, recording_stream=recording_stream)
    robot.visualize_rerun(recording_stream=recording_stream)


    # Move to home pose smoothly
    hz = 60
    n_steps = 40
    for pose in interpolated_poses(robot.get_cartesian_pose(), home_cartesian_pose, n_steps):
        robot.set_cartesian_pose(pose)
        robot.visualize_rerun(recording_stream=recording_stream)
        time.sleep(1.0 / hz)

    # Move to target poses smoothly, visualizing the robot + gripper together
    for i, target in enumerate(target_cartesian_poses):
        for pose in interpolated_poses(robot.get_cartesian_pose(), target, n_steps):
            try:
                robot.set_cartesian_pose(pose)
                robot.visualize_rerun(recording_stream=recording_stream)
                time.sleep(1.0 / hz)
            except ValueError:
                logger.warning("Pose {} unreachable", i + 1)
                exit()

    # Return to home pose smoothly
    for pose in interpolated_poses(robot.get_cartesian_pose(), home_cartesian_pose, n_steps):
        robot.set_cartesian_pose(pose)
        robot.visualize_rerun(recording_stream=recording_stream)
        time.sleep(1.0 / hz)


if __name__ == "__main__":
    main()
