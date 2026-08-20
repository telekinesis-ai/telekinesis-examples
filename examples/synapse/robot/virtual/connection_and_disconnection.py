"""
Connection and disconnection example for the Synapse SDK.

Currently supported only for connecting to real hardware on UR or isaacsim.

Usage:
    python connection_and_disconnection.py [--prim <ROBOT_PRIM_PATH>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(prim: str):
    """Connect to a UR10e at `ip` and cleanly disconnect."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot with given ip
    robot.connect(prim=prim)
    logger.success(f"Connected to UR10e at {prim}.")

    # Sleep for a bit
    time.sleep(2)

    # Disconnect from the robot
    robot.disconnect()
    logger.success("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connection Synapse example")
    parser.add_argument("--prim", type=str, default="/World/ur10e",
                        help="UR robot primitive path in isaacsim")
    args = parser.parse_args()

    main(prim=args.prim)
