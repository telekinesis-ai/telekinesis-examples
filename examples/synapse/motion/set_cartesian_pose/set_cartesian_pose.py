"""
Set Cartesian Pose example for the Synapse SDK.

Drives a real UR10e to the target Cartesian pose.

Currently supported only for real hardware from Universal Robots

For offline, refer to set_cartesian_pose in motion/offline/set_cartesian_pose/

Usage:
    python set_cartesian_pose.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Run the set_cartesian_pose Synapse example."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Define target Cartesian pose
    target_cartesian_pose = [0.5, 0.0, 0.5, 180.0, 0.0, 0.0] 

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
    parser = argparse.ArgumentParser(description="Move the TCP to a target Cartesian pose")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

