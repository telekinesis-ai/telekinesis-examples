"""
Move until Contact example for the Synapse SDK.

Drives a real robot downwards in z direction until contact is detected,
then stops and reports the result.

Currently supported only for Universal Robots (UR10e).

Usage:
    python move_until_contact.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """
    Main function to demonstrate how to create an instance of a robot using the Universal Robots module in Python.
    """

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

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
    # args parser to get ip
    parser = argparse.ArgumentParser(description="UR10e robot move until contact example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
