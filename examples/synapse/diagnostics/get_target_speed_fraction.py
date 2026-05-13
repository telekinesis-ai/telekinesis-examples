"""
Read target speed fraction example for the Synapse SDK.

``get_target_speed_fraction`` returns the **target/desired** speed fraction
set by the operator — either via the teach-pendant speed slider 

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_target_speed_fraction.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the target speed fraction [0.0, 1.0]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        fraction = robot.get_target_speed_fraction()
        logger.success(f"Target speed fraction: {fraction:.3f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target speed fraction Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
