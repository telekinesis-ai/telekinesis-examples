"""
Trigger protective stop example for the Synapse SDK.

Currently supported only for real hardware from Universal Robots

Immediately halts all motion and puts the robot into a protective stop
state. The robot remains powered but frozen until the stop is
acknowledged and cleared from the teach pendant.

Usage:
    python trigger_protective_stop.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Trigger a protective stop on the controller."""

    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Trigger the protective stop and report
    try:
        robot.trigger_protective_stop()
        logger.success("Protective stop triggered.")

    # Ensure we disconnect even if there was an error
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger a protective stop on the robot controller")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

