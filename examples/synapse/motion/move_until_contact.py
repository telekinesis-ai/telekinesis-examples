"""
Move until Contact example for the Synapse SDK.

Drives a real robot downwards in z direction until contact is detected,
then stops and reports the result.

Currently supported only for real hardware from Universal Robots

Usage:
    python move_until_contact.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Move the TCP down in -Z until contact is detected, then report and disconnect."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=args.ip)

    # Move by the cartesian velocity until contact
    contacted = robot.move_until_contact(
        cartesian_velocity=[0, 0, -0.02, 0, 0, 0],
        direction=[0, 0, 0, 0, 0, 0],
        acceleration=0.1,
    )

    # Stop when the robot contact
    if contacted is True:
        logger.info(f"Robot is contacted: {contacted}")
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

