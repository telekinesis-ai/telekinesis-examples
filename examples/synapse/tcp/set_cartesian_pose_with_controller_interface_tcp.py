"""
Example: move 2 cm along -Y from the current pose, first with the default
(tool0) TCP active and then with the controller-interface TCP active.

Requires hardware -- connects to the robot at ``--ip`` and executes the motion,
with live visualization in Rerun.

Demonstrates:
  - ``robot.active_tcp``           -- switch the active end-effector frame
  - ``robot.get_cartesian_pose()`` -- read the current world-frame pose
  - ``robot.set_cartesian_pose()`` -- move via world-frame target; verify arrival

Usage:
    python set_cartesian_pose_with_controller_interface_tcp.py [--ip <ROBOT_IP>]
"""

import argparse
import time

import numpy as np
import rerun as rr
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots



def main(ip: str | None = None):
    # Create the robot and connect to the controller.
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.info(f"Active TCP before add_tcp(): {robot.active_tcp}"
                    f" \nActive TCP transform: {robot.get_active_tcp_transform()}"
                    f" \n TCP pose: {robot.get_cartesian_pose()}")

        current_tcp_pose = robot.get_cartesian_pose()
        target_tcp_pose = list(current_tcp_pose)
        target_tcp_pose[1] -= 0.02  # Move 2 cm along -Y


        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")

        robot.active_tcp = "controller_interface_tcp"

        robot.set_cartesian_pose(target_tcp_pose)
        logger.success(f"arrived: {robot.get_cartesian_pose()}")


    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move -2 cm in Y with the default and controller-interface TCPs."
    )
    parser.add_argument("--ip", type=str, default="192.168.1.100",
                        help="UR controller IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
