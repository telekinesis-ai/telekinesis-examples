"""
Read combined speed scaling example for the Synapse SDK.

``get_speed_scaling_combined`` returns the **actual effective** speed
scaling applied during motion.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_speed_scaling_combined.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the combined runtime speed scaling [0.0, 1.0]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        combined = robot.get_speed_scaling_combined()
        logger.success(f"Combined speed scaling: {combined:.3f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read combined speed scaling Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
