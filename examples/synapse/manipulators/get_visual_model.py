"""
Read the Pinocchio visual geometry model.

Supports Universal Robots (UR), Epson, and virtual/sim.

Usage:
    python get_visual_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio visual geometry model and log a summary."""

    #===================== Create Robot ==========================================
    robot = universal_robots.UniversalRobotsUR10E(name='UR10e')

    # ==================== Run Skill ============================================
    visual_model = robot.get_visual_model()

    logger.info(f"Number of visual geometries: {len(visual_model.geometryObjects)}")
    for geom in visual_model.geometryObjects:
        logger.info(f"  - {geom.name}")


if __name__ == "__main__":
    main()
