"""
Read joint positions example for the Synapse SDK.

Returns the manipulator's joint positions [deg]. With ``--ip``, reads live
hardware state. Without ``--ip``, reads from the internal commanded cache
(no connection made) and logs a warning.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_joint_positions.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current joint positions [deg]."""

    # Create the robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect if --ip was provided, otherwise read from the commanded cache
    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"joint_positions [deg]: {robot.get_joint_positions()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read joint positions Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
