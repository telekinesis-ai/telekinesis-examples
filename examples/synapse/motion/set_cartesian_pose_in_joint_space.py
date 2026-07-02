"""
Set Cartesian Pose in Joint Space example for the Synapse SDK.

Moves to a target Cartesian pose using a trajectory that is linear in joint
space (joint-space ``moveJ`` with internal IK).

Currently supported only for Universal Robots.


Usage:
    python set_cartesian_pose_in_joint_space.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target Cartesian pose via joint-space motion."""
    parser = argparse.ArgumentParser(
        description="UR10e set_cartesian_pose_in_joint_space example"
    )
    parser.add_argument("--ip", type=str,
                        default="192.168.1.100",
                        help="IP address of the UR robot (default: 192.168.1.100)")
    args = parser.parse_args()

    # Create robot instance
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=args.ip)

    # Target Cartesian pose
    current_cartesian_pose = robot.get_cartesian_pose()
    target_pose = current_cartesian_pose.copy()
    target_pose[2] -= 0.1  # Move 10 cm down in

    try:
        robot.set_cartesian_pose_in_joint_space(
            cartesian_pose=target_pose,
            speed=20,
            acceleration=20,
        )
        logger.info(f"Moved to target Cartesian pose: {target_pose}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
