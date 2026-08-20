"""
Check pose against safety limits example for the Synapse SDK.

``is_pose_within_safety_limits`` checks safety plane limits, TCP orientation
deviation limits, and robot reachability by solving IK on the controller.

Currently supported only for real hardware, and only Universal Robots (UR).

Usage:
    python is_pose_within_safety_limits.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Check the robot's current TCP pose against the controller's safety limits."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        current_cartesian_pose = robot.get_cartesian_pose()
        logger.success(
            f"is_pose_within_safety_limits({current_cartesian_pose}): "
            f"{robot.is_pose_within_safety_limits(current_cartesian_pose)}"
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pose against safety limits Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
