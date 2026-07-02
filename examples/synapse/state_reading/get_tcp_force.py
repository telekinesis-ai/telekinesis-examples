"""
Read TCP wrench (force/torque) example for the Synapse SDK.

Returns the TCP wrench ``[Fx, Fy, Fz (N), Tx, Ty, Tz (N·m)]``. Connects to ``--ip`` (default ``192.168.1.100``) and reads the live state.

Currently supported only for real hardware from Universal Robots.

For offline, refer to set_cartesian_pose in state_reading/offline/

Usage:
    python get_tcp_force.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current TCP wrench [N, N·m]."""

    robot = universal_robots.UniversalRobotsUR10E()

    robot.connect(ip=ip)

    try:
        logger.success(f"tcp_force [N, N·m]: {robot.get_tcp_force()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TCP wrench Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
