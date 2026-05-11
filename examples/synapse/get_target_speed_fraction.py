"""
Read the target speed fraction from the Synapse SDK.

The speed fraction is the scalar [0.0, 1.0] set by the teach-pendant
speed slider — it scales every commanded velocity the controller
executes. 1.0 = full programmed speed, 0.0 = stopped. Useful when
diagnosing why a motion is running slower than expected.

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_target_speed_fraction.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def get_target_speed_fraction(ip: str):
    """Read the target speed fraction (0..1) and report it."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Read the target speed fraction and report the result
    try:
        logger.success(f"Target speed fraction: {robot.get_target_speed_fraction()}")

    # Ensure we disconnect even if there was an error
    finally:
        robot.disconnect()


def main():
    """
    Run the get-target-speed-fraction Synapse example.
    Usage:
        python get_target_speed_fraction.py --ip <ROBOT_IP>
    """
    parser = argparse.ArgumentParser(description="Get target speed fraction Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    # Run the example
    get_target_speed_fraction(ip=args.ip)


if __name__ == "__main__":
    main()
