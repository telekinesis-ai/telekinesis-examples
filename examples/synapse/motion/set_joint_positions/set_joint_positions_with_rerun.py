"""
Set Joint Positions with live Rerun logging example for the Synapse SDK.

Drives a real UR10e to a target joint configuration while a babyros subscriber
redraws the robot in Rerun on each state message.

Currently supported only for real hardware from Universal Robots

For offline, refer to set_joint_positions_with_rerun in motion/offline/set_joint_positions/

Usage:
    python set_joint_positions_with_rerun.py [--ip <ROBOT_IP>]
"""

import argparse
from functools import partial

from loguru import logger
from babyros import node

from telekinesis.synapse.robots.manipulators import universal_robots


def on_state(msg, robot):
    """Redraw the robot in Rerun on each state message.

    ``robot`` is bound via functools.partial so the callback keeps babyros's
    single-argument (msg) signature.
    """
    robot.visualize_rerun()


def main(ip: str):
    """Run the set_joint_positions Synapse example with live Rerun logging."""

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")
    robot.connect(ip=ip)

    # Visualize the robot in Rerun
    robot.visualize_rerun()

    # Target: current joint configuration with the base joint rotated +30 deg
    target_joint_positions = [0, -90, -90, -90, 90, 0]

    # Subscriber to states
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot))

    # Command the move, then disconnect cleanly
    try:
        robot.set_joint_positions(
            joint_positions=target_joint_positions,
            speed=60,
            acceleration=80,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    finally:
        sub.delete()
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move to a target joint configuration with live Rerun visualization")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
