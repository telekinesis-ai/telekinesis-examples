"""
Read publisher rate example for the Synapse SDK.

Returns the measured state/TF publish rate in Hz — a smoothed estimate of
what is actually happening, not a configured target. ``None`` if the robot
has no publisher (constructed without a ``name``), ``0.0`` if the publisher
exists but isn't currently running.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_publisher_hz.py [--ip <ROBOT_IP>]
"""

import argparse
import time

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the measured state/TF publish rate of a named robot."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        time.sleep(1.0)  # let the publisher run for a moment before sampling
        logger.success(f"publisher_hz: {robot.get_publisher_hz()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read publisher rate Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
