"""
Read state timestamp example for the Synapse SDK.

Returns the timestamp of the most recent state update [s since epoch].
With ``--ip``, reads live hardware state. Without ``--ip``, reads from the
internal commanded cache (no connection made) and logs a warning.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_timestamp.py [--ip <ROBOT_IP>]
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str | None = None):
    """Log the timestamp of the most recent state update [s]."""

    robot = universal_robots.UniversalRobotsUR10E()

    if ip is not None:
        robot.connect(ip=ip)
    else:
        logger.warning(
            "No --ip provided; reading offline commanded-cache state, "
            "not live hardware readings."
        )

    try:
        logger.success(f"timestamp [s]: {robot.get_timestamp()}")
    finally:
        if ip is not None:
            robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read state timestamp Synapse example")
    parser.add_argument("--ip", type=str, default=None, help="UR robot IP address (optional)")
    args = parser.parse_args()

    main(ip=args.ip)
