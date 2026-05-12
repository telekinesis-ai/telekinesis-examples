"""
Read TCP wrench (force/torque) example for the Synapse SDK.

Returns the TCP wrench ``[Fx, Fy, Fz (N), Tx, Ty, Tz (N·m)]``. With
``--ip``, reads live hardware state. Without ``--ip``, reads from the
internal commanded cache (no connection made) and logs a warning.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_actual_tcp_force.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the current TCP wrench [N, N·m]."""

    robot = universal_robots.UniversalRobotsUR10E()

    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"tcp_force [N, N·m]: {robot.get_actual_tcp_force()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read TCP wrench Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
