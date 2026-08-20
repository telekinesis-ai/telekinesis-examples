"""
Set Joint Position in Cartesian space (relative) example for the Synapse SDK.

Moves to a target configuration using Cartesian motion derived from joint
positions (as opposed to ``set_joint_positions``, which moves in joint
space). On real hardware this dispatches to the UR ``move_l_fk`` RTDE call.

Currently supported only for real hardware. Works on Universal Robots (UR) and Epson.

Usage:
    python set_joint_position_in_cartesian_space.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str):
    """Move to a target joint configuration along a Cartesian trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')
    robot.connect(ip=ip)

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun()

    # ==================== Run Skill ============================================
    try:
        # Target: current joint configuration with the base joint rotated
        target_joint_positions = robot.get_joint_positions().copy()
        target_joint_positions[0] += 5

        robot.set_joint_position_in_cartesian_space(
            joint_positions=target_joint_positions,
            speed=1.05,
            acceleration=1.4,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move to a target joint configuration along a Cartesian trajectory")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
