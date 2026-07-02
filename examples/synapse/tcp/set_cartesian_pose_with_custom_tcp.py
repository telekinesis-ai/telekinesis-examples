"""
Example: move 2 cm along +X from the current pose, first with the default
(tool0) TCP active and then with a custom TCP active.

Requires hardware -- connects to the robot at ``--ip`` and executes the motion.

Demonstrates:
  - ``robot.get_cartesian_pose()`` -- read the current world-frame pose
  - ``robot.set_cartesian_pose()`` -- move via world-frame target; verify arrival
  - ``robot.add_tcp()``            -- register a custom TCP offset and make it active

Currently supported only for real hardware from Universal Robots.

For an offline version, refer to tcp/offline/set_cartesian_pose_with_custom_tcp.py

Usage:
    python set_cartesian_pose_with_custom_tcp.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots

def main(ip: str | None = None):
    """
    Move to a target pose with the default active TCP,
    then add a custom TCP and move to the same target pose with the new TCP active.
    """
    # Create the robot and connect to the controller.
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.info(f"Active TCP: {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        current_tcp_pose = robot.get_cartesian_pose()
        target_tcp_pose = list(current_tcp_pose)
        target_tcp_pose[0] += 0.02  # Move 2 cm along +X


        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")

        # Add custom tcp
        robot.add_tcp(name='gripper_tcp',
                    transform=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    set_active=True)
        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")


    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move 2 cm in X with the default and a custom TCP."
    )
    parser.add_argument("--ip", type=str, default="192.168.1.100",
                        help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
