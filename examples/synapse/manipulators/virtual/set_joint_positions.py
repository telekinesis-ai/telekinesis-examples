"""
Move the robot to a target joint configuration.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python set_joint_positions.py
"""

from loguru import logger
from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the robot to a target joint configuration."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=True)

    #===================== Prepare Target ==========================================
    target_joint_positions = [0, -90, -90, -90, 90, 0]
    
    # ==================== Run Skill ============================================
    robot.set_joint_positions(
        joint_positions=target_joint_positions,
        speed=60,
        acceleration=80,
        asynchronous=False,
    )
    logger.info(f"Moved to target joint positions: {target_joint_positions}")


if __name__ == "__main__":
    main()
