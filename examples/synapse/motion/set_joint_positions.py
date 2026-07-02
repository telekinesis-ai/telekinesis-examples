"""
Set Joint Positions example for the Synapse SDK.

Drives the robot to a target joint configuration.

Currently supported only for Universal Robots (UR10e) for real hardware.

For offline, refer to offline/

Usage:
    python set_joint_positions.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Move the robot to a target joint configuration."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    try:
        # Target: current joint configuration with the base joint rotated +30 deg
        target_joint_positions = robot.get_joint_positions().copy()
        target_joint_positions[0] += 30

        # Move to target joint positions
        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=60,
            acceleration=80,
            asynchronous=False,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    finally:
        # Disconnect
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

