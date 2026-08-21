"""
Read the Pinocchio collision geometry model.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_collision_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio collision geometry model and log a summary."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    collision_model = robot.get_collision_model()

    logger.info(f"Number of collision geometries: {len(collision_model.geometryObjects)}")
    logger.info(f"Number of collision pairs: {len(collision_model.collisionPairs)}")


if __name__ == "__main__":
    main()
