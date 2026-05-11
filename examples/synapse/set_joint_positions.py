"""
Set Joint Positions example for the Synapse SDK.

Drives the robot to a target joint configuration.

Real hardware is currently supported only for Universal Robots (UR10e).
Offline mode is supported for all manipulator brands — to run offline, omit
the ``robot.connect()`` / ``robot.disconnect()`` calls below; the SDK will
update the commanded-state cache (via FK) without touching hardware.

Usage:
    python set_joint_positions.py --ip <ROBOT_IP>
"""

import argparse
from telekinesis.synapse.robots.manipulators import universal_robots


def main(robot_ip: str):
    """
    Main function to demonstrate how to create an instance of a robot using the Universal Robots module in Python.

    Args:
        robot_ip (str): The IP address of the UR robot
    Returns:
        None
    Raises:
        None
    """

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()

    # Connect to the robot
    robot.connect(ip="192.168.1.2")

    # Move to target joint positions
    robot.set_joint_positions(
        joint_positions=[0, -90, 90, 0, 0, 0],
        speed=60,
        acceleration=80,
        asynchronous=False,
    )

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    # args parser to get ip
    parser = argparse.ArgumentParser(description="UR10e robot set joint positions example")
    parser.add_argument("--ip", type=str, default="192.168.1.2", help="IP address of the UR robot")
    args = parser.parse_args()

    main(args.ip)
