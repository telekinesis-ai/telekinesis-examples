"""
Example: move to 3 world-frame poses using cartesian control, with both a
default and a custom TCP frame -- offline.

No hardware required -- runs entirely on the kinematic model.

Demonstrates:
  - ``robot.attach_tool()``        -- attach a gripper
  - ``robot.add_tcp()``            -- register a custom TCP offset
  - ``robot.set_cartesian_pose()`` -- move via world-frame target
  - ``robot.active_tcp``           -- switch the active end-effector frame

Run:
    python set_cartesian_pose_with_custom_tcp.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse.tools.parallel_grippers import onrobot


def main():

    # Create robot
    robot = universal_robots.UniversalRobotsUR10E()

    # Home pose and target cartesian poses
    home_cartesian_pose = [0.40, 0.00, 0.60, 180.0, 0.0, 0.0]
    target_cartesian_poses = [
        [0.40,  0.10,  0.45, 180.0, 0.0, 0.0],
        [0.40, -0.10,  0.45, 180.0, 0.0, 0.0],
        [0.35,  0.00,  0.55, 180.0, 0.0, 0.0],
    ]

    # Move to home to establish a known starting pose
    robot.set_cartesian_pose(home_cartesian_pose)

    # Section 1: cartesian control with the default (tool0) TCP active
    logger.info(f"--- Section 1: cartesian control (active TCP: '{robot.active_tcp}') ---")
    for target in target_cartesian_poses:
        robot.set_cartesian_pose(target)
        logger.info(f"Moved to {target}")
    robot.set_cartesian_pose(home_cartesian_pose)

    # Attach a gripper and add a custom TCP at its tip
    gripper = onrobot.OnRobotRG6()
    robot.attach_tool(gripper)
    robot.add_tcp(name="gripper_tip",
                  transform=[0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                  set_active=True)

    # Section 2: cartesian control with the gripper tip as the active TCP
    logger.info(f"--- Section 2: cartesian control (active TCP: '{robot.active_tcp}') ---")
    for target in target_cartesian_poses:
        robot.set_cartesian_pose(target)
        logger.info(f"Moved to {target}")
    robot.set_cartesian_pose(home_cartesian_pose)


if __name__ == "__main__":
    main()
