"""
Check live connection status example for the Synapse SDK.

``is_connected`` reports whether the manipulator state is being driven by
live hardware. Returns ``False`` before ``connect()`` and after
``disconnect()``; returns ``True`` while a live RTDE/equivalent session
is open. Currently supported only for Universal Robots.

With ``--ip``, logs the value before connect, after connect, and after
disconnect. Without ``--ip``, only the offline (``False``) case is shown.

Usage:
    python is_connected.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log ``is_connected`` offline, and around a connect/disconnect cycle if ``ip`` is given."""

    # Create the robot instance (no hardware yet)
    robot = universal_robots.UniversalRobotsUR10E()
    logger.info(f"is_connected (pre-connect): {robot.is_connected()}")

    if ip is None:
        return

    # Connect to the robot with given ip
    robot.connect(ip=ip)
    logger.success(f"is_connected (post-connect): {robot.is_connected()}")

    # Disconnect from the robot
    robot.disconnect()
    logger.info(f"is_connected (post-disconnect): {robot.is_connected()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="is_connected Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
