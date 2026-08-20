"""
Read target (commanded) TCP pose example for the Synapse SDK — offline.

Returns the target/commanded TCP pose ``[x, y, z (m), rx, ry, rz (deg)]``. Reads from the internal commanded-cache
state; no hardware connection is made.

Illustrated using Universal Robots (UR10e), supported on all robots.

Usage:
    python get_target_tcp_pose_offline.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log the commanded-cache target TCP pose [m, deg]."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E()

    # ==================== Run Skill ============================================
    logger.success(f"target_tcp_pose [m, deg]: {robot.get_target_tcp_pose()}")


if __name__ == "__main__":
    main()
