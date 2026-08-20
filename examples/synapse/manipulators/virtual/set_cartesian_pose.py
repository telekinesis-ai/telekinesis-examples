"""
Set Cartesian Pose example for the Synapse SDK -- offline.

Moves the TCP to a target Cartesian pose on the kinematic model; no hardware
connection is made.

Supports all robots.

Usage:
    python set_cartesian_pose.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Move the TCP to a target Cartesian pose on the kinematic model."""

    # ===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Visualization (Optional) =============================
    robot.visualize_rerun()

    # ==================== Run Skill ============================================
    # Define target Cartesian pose
    target_cartesian_pose = [0.5, 0.0, 0.5, 180.0, 0.0, 0.0]

    # Command the move
    robot.set_cartesian_pose(
        cartesian_pose=target_cartesian_pose,
        speed=0.1,
        acceleration=0.1,
    )
    logger.info(f"Moved to target Cartesian pose: {target_cartesian_pose}")


if __name__ == "__main__":
    main()
