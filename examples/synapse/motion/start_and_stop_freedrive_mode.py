"""
Freedrive mode example for the Synapse SDK.

``start_freedrive_mode`` puts the robot into hand-guiding mode — the operator
can physically push the arm and it complies. ``free_axes`` is a 6-element
mask ``[x, y, z, rx, ry, rz]`` where ``1`` means the axis is free and ``0``
means it is locked. ``stop_freedrive_mode`` returns the controller to normal
motion control.

Currently supported only for Universal Robots (UR10e).

Usage:
    python freedrive_mode.py --ip <ROBOT_IP>
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """Enter freedrive for 10 seconds, then exit."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip=robot_ip)

    # Enter freedrive with all axes free
    free_axes = [1, 1, 1, 1, 1, 1]
    logger.info(f"Starting freedrive - free axes: {free_axes}")
    robot.start_freedrive_mode(free_axes=free_axes)

    # Hold freedrive open for hand-guiding
    time.sleep(10)

    # Exit freedrive
    robot.stop_freedrive_mode()
    logger.success("Freedrive mode stopped.")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR robot freedrive mode example")
    parser.add_argument("--ip", type=str, required=True, help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
