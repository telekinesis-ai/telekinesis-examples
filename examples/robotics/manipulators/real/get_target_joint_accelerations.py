"""
Logs the controller-commanded target joint accelerations.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_target_joint_accelerations.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Log the controller-commanded target joint accelerations [deg/s²]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        # ==================== Run Skill ============================================
        logger.success(f"target_joint_accelerations [deg/s^2]: {robot.get_target_joint_accelerations()}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target joint accelerations Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
