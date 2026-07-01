"""
Connection and disconnection example for the Synapse SDK.

Connects to and disconnects from real robot hardware. Currently supported
only for Universal Robots (UR10e).

Usage:
    python connection_and_disconnection.py [--ip <ROBOT_IP>]
"""

import argparse
import time
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Connect to a UR10e at `ip` and cleanly disconnect."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot with given ip
    robot.connect(ip=ip)
    logger.success(f"Connected to UR10e at {ip}.")

    # Sleep for a bit
    time.sleep(2)

    # Disconnect from the robot
    robot.disconnect()
    logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connection Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
