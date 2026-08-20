"""
Read target (commanded) joint velocities example for the Synapse SDK — offline.

Returns the manipulator's target/commanded joint velocities [deg/s]. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_target_joint_velocities_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target joint velocities [deg/s]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    logger.success(f"target_joint_velocities [deg/s]: {robot.get_target_joint_velocities()}")


if __name__ == "__main__":
    main()
