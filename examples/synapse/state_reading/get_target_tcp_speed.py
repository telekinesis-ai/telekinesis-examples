"""
Read target (commanded) TCP velocity example for the Synapse SDK.

Returns the target/commanded TCP twist
``[vx, vy, vz (m/s), ωx, ωy, ωz (deg/s)]``. Connects to ``--ip`` (default ``192.168.1.100``) and reads the live state.

Currently supported only for real hardware from Universal Robots.

For offline, refer to set_cartesian_pose in state_reading/offline/

Usage:
    python get_target_tcp_speed.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current target TCP velocity [m/s, deg/s]."""

    robot = universal_robots.UniversalRobotsUR10E()

    robot.connect(ip=ip)

    try:
        logger.success(f"target_tcp_speed [m/s, deg/s]: {robot.get_target_tcp_speed()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target TCP velocity Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
