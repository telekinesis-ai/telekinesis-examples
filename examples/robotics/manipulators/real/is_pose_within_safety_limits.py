"""
Checks safety plane limits, TCP orientation deviation limits, and reachability for a Cartesian pose.

Supports Universal Robots (UR).

Usage:
    python is_pose_within_safety_limits.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Check the robot's current TCP pose against the controller's safety limits."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        current_cartesian_pose = robot.get_cartesian_pose()
        logger.success(
            f"is_pose_within_safety_limits({current_cartesian_pose}): "
            f"{robot.is_pose_within_safety_limits(current_cartesian_pose)}"
        )
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pose against safety limits Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
