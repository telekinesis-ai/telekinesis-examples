"""
Read robot status example for the Synapse SDK.

Returns the controller's high-level robot status (e.g. ``"NORMAL"``,
``"REDUCED"``, ``"PROTECTIVE_STOP"``).

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_robot_status.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the current robot status."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.success(f"Robot status: {robot.get_robot_status()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read robot status Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
