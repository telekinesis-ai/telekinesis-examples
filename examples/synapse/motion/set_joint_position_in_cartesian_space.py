"""
Set Joint Position in Cartesian Space example for the Synapse SDK.

Moves to a target joint configuration using a trajectory that is linear in
Cartesian space.

Currently supported only for real hardware from Universal Robots

Usage:
    python set_joint_position_in_cartesian_space.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Move to a target joint configuration via Cartesian-space motion."""
    
    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # Target joint positions in degrees
    q_target = [0, -90, 0, -90, 0, 0]

    # Safety warning before commanding real motion
    logger.warning(
        f"About to move real robot to joint positions {q_target}. "
        "Make sure it's safe to move there, otherwise use the advanced example."
    )

    try:
        robot.set_joint_position_in_cartesian_space(
            joint_positions=q_target,
        )
        logger.info(f"Moved to target joint positions: {q_target}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool contact polling Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)

