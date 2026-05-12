"""
Read target (commanded) joint positions example for the Synapse SDK.

Returns the manipulator's target/commanded joint positions [deg]. With
``--ip``, reads live hardware state. Without ``--ip``, reads from the
internal commanded cache (no connection made) and logs a warning.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_target_joint_positions.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current target joint positions [deg]."""

    robot = universal_robots.UniversalRobotsUR10E()

    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"target_joint_positions [deg]: {robot.get_target_joint_positions()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target joint positions Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
