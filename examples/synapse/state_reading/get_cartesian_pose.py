"""
Read TCP Cartesian pose example for the Synapse SDK.

Returns the TCP pose ``[x, y, z (m), rx, ry, rz (deg)]``. With ``--ip``,
reads live hardware state. Without ``--ip``, reads from the internal
commanded cache (no connection made) and logs a warning.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_cartesian_pose.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current TCP pose [m, deg]."""

    robot = universal_robots.UniversalRobotsUR10E()

    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"tcp_pose [m, deg]: {robot.get_cartesian_pose()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TCP Cartesian pose Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
