"""
Read robot mode example for the Synapse SDK.

Returns the controller's high-level robot mode (e.g. ``"RUNNING"``,
``"IDLE"``, ``"POWER_OFF"``).

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_robot_mode.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the current robot mode."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        logger.success(f"Robot mode: {robot.get_robot_mode()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read robot mode Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
