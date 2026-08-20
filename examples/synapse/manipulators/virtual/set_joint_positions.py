"""
Set Joint Positions example for the Synapse SDK -- offline.

Moves the robot to a target joint configuration on the kinematic model; no
hardware connection is made.

Supports all robots.

Usage:
    python set_joint_positions.py
"""

from loguru import logger
from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the robot to a target joint configuration."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    # Target: current joint configuration with the base joint rotated +30 deg
    target_joint_positions = [0, -90, -90, -90, 90, 0]

    # Move to target joint positions
    robot.set_joint_positions(
        joint_positions=target_joint_positions,
        speed=60,
        acceleration=80,
        asynchronous=False,
    )
    logger.info(f"Moved to target joint positions: {target_joint_positions}")

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun(live=False)


if __name__ == "__main__":
    main()
