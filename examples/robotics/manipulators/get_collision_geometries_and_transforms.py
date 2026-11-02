"""
Place the robot's collision geometries in the world frame for a joint configuration.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_collision_geometries_and_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Log each collision geometry of the robot and its world-frame transform."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    q = [0, -90, 90, 0, 90, 0]
    geometries, transforms = robot.get_collision_geometries_and_transforms(joint_positions=q)
    logger.info(f"Collision geometries: {len(geometries)}")

    for geometry, transform in zip(geometries, transforms):
        logger.info(f"{geometry}:\n{transform}")

    robot.shutdown()


if __name__ == "__main__":
    main()
