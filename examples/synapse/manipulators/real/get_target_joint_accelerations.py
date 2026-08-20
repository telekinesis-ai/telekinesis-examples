"""
Read target joint accelerations example for the Synapse SDK.

Returns the target joint accelerations from the controller [deg/s²]. Zero
when the backend does not report a commanded acceleration. Reads from
``self.state``.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_target_joint_accelerations.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the controller-commanded target joint accelerations [deg/s²]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"target_joint_accelerations [deg/s^2]: {robot.get_target_joint_accelerations()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target joint accelerations Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
