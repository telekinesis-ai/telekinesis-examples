"""
Trigger protective stop example for the Synapse SDK.

Currently supported only for Universal Robots (UR10e).

Immediately halts all motion and puts the robot into a protective stop
state. The robot remains powered but frozen until the stop is
acknowledged and cleared from the teach pendant.

Note: requires an active motion program on the controller. Calling this
on an idle robot raises "RTDE control script is not running".

Usage:
    python trigger_protective_stop.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Trigger a protective stop on the controller."""

    # Create and connect to the robot
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
    parser = argparse.ArgumentParser(description="Trigger protective stop Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    main(ip=args.ip)
