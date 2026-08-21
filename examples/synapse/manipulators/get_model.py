"""
Read the Pinocchio kinematic model.

Supports Universal Robots (UR), Epson, and virtual.

Usage:
    python get_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio kinematic model and log a few summary fields."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    model = robot.get_model()
    logger.success(f"Model: {model}")
    logger.info(f"Model name: {model.name}")
    logger.info(f"nq (configuration dim): {model.nq}")
    logger.info(f"nv (velocity dim): {model.nv}")
    logger.info(f"Number of joints: {model.njoints}")
    logger.info(f"Number of frames: {model.nframes}")


if __name__ == "__main__":
    main()
