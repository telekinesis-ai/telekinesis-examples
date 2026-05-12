"""
Read the Pinocchio collision geometry model for the Synapse SDK.

``get_collision_model`` returns the Python Pinocchio ``GeometryModel``
populated with the robot's collision geometries from the URDF. Useful for
custom collision-checking pipelines built on top of Pinocchio / hpp-fcl.

Universal Robots (UR10e) is used here purely for illustration.
This example runs purely on the kinematic model and does not connect to
hardware — no ``--ip`` is required.

Usage:
    python get_collision_model.py
"""

from loguru import logger

from telekinesis.synapse.robots.manipulators import universal_robots


def main():
    """Read the Pinocchio collision geometry model and log a summary."""

    # Create the robot (no connect required — runs on the kinematic model)
    robot = universal_robots.UniversalRobotsUR10E()

    # Read the Pinocchio collision GeometryModel
    collision_model = robot.get_collision_model()
    
    logger.info(f"Number of collision geometries: {len(collision_model.geometryObjects)}")
    logger.info(f"Number of collision pairs: {len(collision_model.collisionPairs)}")


if __name__ == "__main__":
    main()
