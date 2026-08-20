"""
Read target joint velocities example for the Synapse SDK.

Returns the controller-commanded target joint velocities [deg/s]. Zero when
the backend does not report a commanded velocity. Reads from ``self.state``.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_target_joint_velocities.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the controller-commanded target joint velocities [deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        logger.success(f"target_joint_velocities [deg/s]: {robot.get_target_joint_velocities()}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read target joint velocities Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
