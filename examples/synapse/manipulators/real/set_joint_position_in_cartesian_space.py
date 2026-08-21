"""
Moves to a target joint configuration along a Cartesian motion trajectory.

Supports Universal Robots (UR) and virtual.

Usage:
    python set_joint_position_in_cartesian_space.py [--ip <ROBOT_IP>]
"""

import argparse

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main(ip: str) -> None:
    """Move to a target joint configuration along a Cartesian trajectory."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    # Live: subscribes to the robot's state topic and redraws as it moves.
    robot.visualize_rerun(live=True)

    try:
        #===================== Connect Robot ==========================================
        robot.connect(ip=ip)

        #===================== Prepare Target ==========================================
        # Target: current joint configuration with the base joint rotated
        target_joint_positions = robot.get_joint_positions().copy()
        target_joint_positions[0] += 5

        # ==================== Run Skill ============================================
        robot.set_joint_position_in_cartesian_space(
            joint_positions=target_joint_positions,
            speed=1.05,
            acceleration=1.4,
        )
        logger.info(f"Moved to target joint positions: {target_joint_positions}")
    except (ConnectionError, OSError) as e:
        logger.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move to a target joint configuration along a Cartesian trajectory")
    parser.add_argument("--ip", type=str, default="192.168.1.100", help="UR robot IP address (default: 192.168.1.100)")
    args = parser.parse_args()

    main(ip=args.ip)
