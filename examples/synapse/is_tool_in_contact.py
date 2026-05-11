"""
Tool contact example for the Synapse SDK.

This example runs against real robot hardware. Currently supported only
for Universal Robots (UR10e).

Usage:
    python is_tool_in_contact.py --ip <ROBOT_IP>
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def tool_contact_example(ip: str):
    """Check whether the tool is contacting in the +Z direction (tool frame).

    Expected behavior: nothing moves — pure query. The wrist sensor reports
    whether contact is detected along +Z right now. With nothing touching
    the TCP, expect False/0; press the TCP and expect True/1.
    """

    # Create and connect to the robot
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Check tool contact along +Z and report the result
    try:
        direction = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        contact = robot.is_tool_in_contact(direction=direction)
        logger.success(f"Tool contact: {contact}")

    # Ensure we stop the robot and disconnect even if there was an error
    finally:
        robot.disconnect()


def main():
    """
    Run the tool contact Synapse example.
    Usage:
        python is_tool_in_contact.py --ip <ROBOT_IP>
    """
    parser = argparse.ArgumentParser(description="Tool contact Synapse example")
    parser.add_argument("--ip", type=str, required=True, help="UR robot IP address")
    args = parser.parse_args()

    # Run the example
    tool_contact_example(ip=args.ip)


if __name__ == "__main__":
    main()
