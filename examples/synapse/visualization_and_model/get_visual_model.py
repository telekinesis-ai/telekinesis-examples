"""
Read the Pinocchio visual geometry model for the Synapse SDK.

``get_visual_model`` returns the Python Pinocchio ``GeometryModel``
populated with the robot's visual geometries from the URDF. 

Universal Robots (UR10e) is used here purely for illustration. It supports all robots.

Usage:
    python get_visual_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio visual geometry model and log a summary."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read the Pinocchio visual GeometryModel
    visual_model = robot.get_visual_model()
    
    logger.info(f"Number of visual geometries: {len(visual_model.geometryObjects)}")
    for geom in visual_model.geometryObjects:
        logger.info(f"  - {geom.name}")


if __name__ == "__main__":
    main()
