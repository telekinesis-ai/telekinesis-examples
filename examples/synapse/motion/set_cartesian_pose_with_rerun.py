"""
Set Cartesian Pose with live Rerun logging example for the Synapse SDK.

Drives a real UR10e to the target Cartesian pose while a babyros subscriber
redraws the robot in Rerun on each state message.

Real robots are currently supported only for Universal Robots for real hardware.

For offline, refer to set_cartesian_pose_with_rerun in motion/offline/

Usage:
    python set_cartesian_pose_with_rerun.py [--ip <ROBOT_IP>]
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


def main():
    """Run the set_cartesian_pose Synapse example."""
    parser = argparse.ArgumentParser(description="UR10e set_cartesian_pose example")
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

    # Define target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_cartesian_pose = current_cartesian_pose.copy()
    target_cartesian_pose[2] -= 0.2  # Move 20 cm down in Z

    # Subscriber to states
    sub = node.Subscriber(topic=robot.state_publisher_topic,
                          callback=partial(on_state, robot=robot))

    # Command the move, then disconnect cleanly
    try:
        robot.set_cartesian_pose(
            cartesian_pose=target_cartesian_pose,
            speed=0.1,
            acceleration=0.1,
        )
        logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")
    finally:
        sub.delete()
        robot.disconnect()

if __name__ == "__main__":
    main()
