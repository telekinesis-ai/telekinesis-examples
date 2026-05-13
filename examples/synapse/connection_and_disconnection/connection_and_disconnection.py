"""
Connection and disconnection example for the Synapse SDK.

Connection and disconnection are to real robot hardware. Currently supported only
for Universal Robots.

Usage:
    python connection_and_disconnection.py --ip <ROBOT_IP>
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
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
