"""
Check live connection status example for the Synapse SDK.

``is_connected`` reports whether the manipulator state is being driven by
live hardware.

Connects to ``--ip`` (default ``192.168.1.100``) and logs the value before
connect, after connect, and after disconnect.

Currently supported only for real hardware from Universal Robots.

For offline, refer to set_cartesian_pose in state_reading/offline/

Usage:
    python is_connected.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log ``is_connected`` before, during, and after a connect/disconnect cycle."""

    # Create the robot instance (no hardware yet)
    robot = universal_robots.UniversalRobotsUR10E()
    logger.info(f"is_connected (pre-connect): {robot.is_connected()}")

    # Connect to the robot with given ip
    robot.connect(ip=ip)
    logger.success(f"is_connected (post-connect): {robot.is_connected()}")

    # Disconnect from the robot
    robot.disconnect()
    logger.info(f"is_connected (post-disconnect): {robot.is_connected()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="is_connected Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
