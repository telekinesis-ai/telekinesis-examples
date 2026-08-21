"""
Read per-link visual mesh world transforms for a manipulator.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_visual_mesh_transforms.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots
from telekinesis.synapse import utils


def main():
    """Read every link's visual-mesh world transform at the current joint configuration."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    transforms = robot.get_visual_mesh_transforms()
    logger.info(f"Number of links with visual meshes: {len(transforms)}")

    # Convert each 4x4 transformation matrix to a pose [x, y, z, rx, ry, rz] (m, deg)
    for name, T in transforms.items():
        pose = utils.transformation_matrix_to_pose(T, rot_type="deg")
        logger.success(f"{name}: pose [m, deg] = {pose}")


if __name__ == "__main__":
    main()
