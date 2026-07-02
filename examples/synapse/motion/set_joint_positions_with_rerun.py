"""
Set Joint Positions with live Rerun logging example for the Synapse SDK.

Drives a real robot to a target joint configuration while a babyros subscriber
logs each state message and redraws the robot in Rerun.

Real robots are currently supported only for Universal Robots.

For offline, refer to set_joint_positions_with_rerun in motion/offline/

Usage:
    python set_joint_positions_with_rerun.py [--ip <ROBOT_IP>]
"""

import argparse
from functools import partial

from loguru import logger
from babyros import node

from telekinesis.synapse.robots.manipulators import universal_robots


def on_state(msg, robot):
    """Log each state message and redraw the robot in Rerun.

    ``robot`` is bound via functools.partial so the callback keeps babyros's
    single-argument (msg) signature.
    """
    logger.info(f"Received robot state: {msg}")
    robot.visualize_rerun()


def main():
    """Run the set_joint_positions Synapse example with live Rerun logging."""
    parser = argparse.ArgumentParser(description="UR10e set_joint_positions example")
    parser.add_argument("--ip", type=str,
                        default="192.168.1.100",
                        help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create robot instance with a name so its state publisher starts
    # (the subscriber below needs a non-empty state_publisher_topic).
    robot = universal_robots.UniversalRobotsUR10E(name="manipulator1")
    robot.connect(ip=args.ip)

    # Visualize the robot in Rerun
    robot.visualize_rerun()

    # Target: current joint configuration with the base joint rotated +30 deg
    target_joint_positions = robot.get_joint_positions().copy()
    target_joint_positions[0] += 30

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
    main()
