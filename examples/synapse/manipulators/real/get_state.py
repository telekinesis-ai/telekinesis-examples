"""
Read the full robot state dictionary example for the Synapse SDK.

Returns the same state dict broadcast over the robot's state topic, with keys
such as ``joint_positions``, ``joint_velocities``, ``tcp_pose``,
``target_joint_positions``, ``target_tcp_pose`` and ``timestamp`` (plus
hardware-dependent optional fields).

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python get_state.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Log the full live robot state dictionary."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        state = robot.get_state()
        for key, value in state.items():
            logger.success(f"{key}: {value}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read full robot state Synapse example")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
