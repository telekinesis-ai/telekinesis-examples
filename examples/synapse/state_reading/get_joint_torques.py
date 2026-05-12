"""
Read joint torques example for the Synapse SDK.

Returns the manipulator's joint torques [N·m]. With ``--ip``, reads live
hardware state. Without ``--ip``, reads from the internal commanded cache
(no connection made) and logs a warning.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_joint_torques.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current joint torques [N·m]."""

    robot = universal_robots.UniversalRobotsUR10E()

    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"joint_torques [N·m]: {robot.get_joint_torques()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read joint torques Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
