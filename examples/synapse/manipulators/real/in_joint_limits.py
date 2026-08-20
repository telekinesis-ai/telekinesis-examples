"""
Check joint limits example for the Synapse SDK.

``in_joint_limits`` checks whether a joint configuration lies within the
limits derived from the robot's URDF. This is a kinematic check only — it
does not read from or move the connected robot.

Supported for all robots, in every connection mode.

Usage:
    python in_joint_limits.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Check the robot's current joint configuration and an out-of-range one."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()
    robot.connect(ip=ip)

    # ==================== Run Skill ============================================
    try:
        current_joint_positions = robot.get_joint_positions()
        logger.success(
            f"Current joint positions within limits: "
            f"{robot.in_joint_limits(q=current_joint_positions, verbose=True)}"
        )

        out_of_range = robot.joint_limits[:, 1] + 10.0  # 10 deg past every upper limit
        logger.info(
            f"Configuration past the upper limits within limits: "
            f"{robot.in_joint_limits(q=out_of_range, verbose=True)}"
        )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a joint configuration against the robot's limits")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
