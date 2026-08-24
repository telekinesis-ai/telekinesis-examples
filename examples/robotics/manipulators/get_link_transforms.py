"""
Read per-link world transforms for a manipulator.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_link_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse import utils


def main():
    """Read every link's world transform at the current joint configuration."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    transforms = robot.get_link_transforms()
    logger.info(f"Number of frames: {len(transforms)}")

    # Convert each 4x4 transformation matrix to a pose [x, y, z, rx, ry, rz] (m, deg)
    for name, T in transforms.items():
        pose = utils.transformation_matrix_to_pose(T, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")


if __name__ == "__main__":
    main()
