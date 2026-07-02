"""
Set Cartesian Pose example for the Synapse SDK.

Drives a real UR10e to the target Cartesian pose.

Currently supported only for real hardware from Universal Robots

For offline, refer to set_cartesian_pose in motion/offline/

Usage:
    python set_cartesian_pose.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Run the set_cartesian_pose Synapse example."""
    parser = argparse.ArgumentParser(description="UR10e set_cartesian_pose example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # Define target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] += 0.1  # Move 10 cm up in Z

    # Command the move, then disconnect cleanly
    try:
        robot.set_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            speed=0.1,
            acceleration=0.1,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
