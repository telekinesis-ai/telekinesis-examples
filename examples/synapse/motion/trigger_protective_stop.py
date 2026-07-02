"""
Trigger protective stop example for the Synapse SDK.

Currently supported only for Universal Robots.

Immediately halts all motion and puts the robot into a protective stop
state. The robot remains powered but frozen until the stop is
acknowledged and cleared from the teach pendant.

Usage:
    python trigger_protective_stop.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Trigger a protective stop on the controller."""

    parser = argparse.ArgumentParser(description="Trigger protective stop Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # Trigger the protective stop and report
    try:
        robot.trigger_protective_stop()
        logger.success("Protective stop triggered.")

    # Ensure we disconnect even if there was an error
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
