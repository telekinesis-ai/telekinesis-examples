"""
Set Cartesian Pose (relative) example for the Synapse SDK.

Reads the current TCP pose and moves to a target defined *relative* to it
(an offset applied to the current pose).

Currently supported only for real hardware from Universal Robots.

For an offline version, refer to set_cartesian_pose_relative in motion/offline/set_cartesian_pose/

Usage:
    python set_cartesian_pose_relative.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Run the set_cartesian_pose Synapse example."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Define a target relative to the current pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

    # Command the move, then disconnect cleanly
    try:
        robot.set_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            speed=0.5,
            acceleration=0.5,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move the TCP to a target Cartesian pose")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

