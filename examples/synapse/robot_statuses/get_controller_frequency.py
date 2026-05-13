"""
Read controller frequency example for the Synapse SDK.

``get_controller_frequency`` measures the controller's update rate by
polling ``get_timestamp()`` for ``window_s`` seconds and computing
``1 / mean_step_time``. 

Currently supported only for Universal Robots (UR10e).

Usage:
    python get_controller_frequency.py --ip <ROBOT_IP>
"""

import argparse
from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the measured controller update frequency [Hz]."""

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        # window_s defaults to 0.2 s; pass explicitly for clarity
        frequency = robot.get_controller_frequency(window_s=0.2)
        logger.success(f"Controller frequency [Hz]: {frequency:.2f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read controller frequency Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
